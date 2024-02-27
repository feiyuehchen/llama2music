# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import time

import torch
from transformers import LlamaTokenizer

from llama_recipes.inference.safety_utils import get_safety_checker, AgentType
from llama_recipes.inference.model_utils import load_model, load_peft_model
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)

from tqdm import tqdm
import copy
import json
from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG

from torch.utils.data import Dataset

from datasets import load_dataset, load_from_disk

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
def get_dataset(data_path):
    dataset = load_from_disk(data_path)["test"]
    prompts = []
    for data in dataset:

        if data.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(data)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(data)
        prompts.append([prompt, data["dataset_type"]])

    return prompts

def main(
    model_name: str="/home/feiyuehchen/personality/llama/llama-2-7b-hf",
    peft_model: str="/home/feiyuehchen/personality/llama2music/scripts/training/PATH/to/save/PEFT/model",
    quantization: bool=True,
    save_path: str="OUTPUT/REMI_20000_model_2048.json",
    max_new_tokens = 2048, #The maximum numbers of tokens to generate
    prompt_file: str="/home/feiyuehchen/personality/llama2music/dataset/processed_data",
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=False,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.2, # [optional] The value used to modulate the next token probabilities.
    top_k: int=10, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool=True, # Enable safety check with Salesforce safety flan t5
    enable_llamaguard_content_safety: bool=False,
    max_padding_length: int=512, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):
    # if prompt_file is not None:
    #     assert os.path.exists(
    #         prompt_file
    #     ), f"Provided Prompt file does not exist {prompt_file}"
    #     with open(prompt_file, "r") as f:
    #         user_prompt = "\n".join(f.readlines())
    # elif not sys.stdin.isatty():
    #     user_prompt = "\n".join(sys.stdin.readlines())
    # else:
    #     print("No user prompt provided. Exiting.")
    #     sys.exit(1)
    # Update the configuration for the training and sharding process
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)
    dataset_config = generate_dataset_config(train_config, kwargs)

    

    # safety_checker = get_safety_checker(enable_azure_content_safety,
    #                                     enable_sensitive_topics,
    #                                     enable_salesforce_content_safety,
    #                                     enable_llamaguard_content_safety
    #                                     )

    # # Safety check of the user prompt
    # safety_results = [check(user_prompt) for check in safety_checker]
    # are_safe = all([r[1] for r in safety_results])
    # if are_safe:
    #     print("User prompt deemed safe.")
    #     print(f"User prompt:\n{user_prompt}")
    # else:
    #     print("User prompt deemed unsafe.")
    #     for method, is_safe, report in safety_results:
    #         if not is_safe:
    #             print(method)
    #             print(report)
    #     print("Skipping the inference as the prompt is not safe.")
    #     sys.exit(1)  # Exit the program with an error status

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    model = load_model(model_name, quantization)

    tokenizer = LlamaTokenizer.from_pretrained(peft_model)
    tokenizer.pad_token = tokenizer.eos_token

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()
    
    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)    
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")



    # dataset_val = get_preprocessed_dataset(
    #         tokenizer,
    #         dataset_config,
    #         split="test",
    #     )
    # val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")
    # test_dataloader = torch.utils.data.DataLoader(
    #     dataset_val,
    #     num_workers=train_config.num_workers_dataloader,
    #     pin_memory=True,
    #     **val_dl_kwargs,
    # )


    prompts = get_dataset(prompt_file)

    output_json = []   
    with torch.no_grad(): 
        for user_prompt, dataset_type in tqdm(prompts):
        
            batch = tokenizer(user_prompt, padding="max_length", truncation=True, max_length=max_padding_length, return_tensors="pt")
            batch = {k: v.to("cuda") for k, v in batch.items()}
            # start = time.perf_counter()
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs 
            )

            # e2e_inference_time = (time.perf_counter()-start)*1000
            # print(f"the inference time is {e2e_inference_time} ms")
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # print(f"Model output:\n{output_text}")

            output_json.append({
                "input": user_prompt,
                "output": output_text,
                "dataset_type": dataset_type
            })
            
            os.makedirs(os.path.dirname(save_path), exist_ok = True)
            with open(save_path, 'w') as f:
                json.dump(output_json, f, indent=4)

    

    
    
    


if __name__ == "__main__":
    fire.Fire(main)
