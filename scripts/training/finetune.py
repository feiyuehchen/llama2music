import torch
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer,
    default_data_collator, 
    Trainer, 
    TrainingArguments,
    TrainerCallback
)
from datasets import load_dataset, load_from_disk
from llama_recipes.configs import fsdp_config, train_config
import argparse
import torch
from torch.utils.data import Dataset
import copy

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_int8_training,
)

from contextlib import nullcontext
from build_dataset import build_instruction_dataset, DataCollatorForSupervisedDataset

from llama_recipes.utils.dataset_utils import get_preprocessed_dataset
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)

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

# lr=1e-4
# lora_rank=48
# lora_alpha=96
# lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
# modules_to_save="embed_tokens,lm_head"
# lora_dropout=0.05

class MusicDataset(Dataset):
    def __init__(self, data_dir, tokenizer, split):
        self.ann = load_from_disk(data_dir)[split]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss


        ann = self.ann[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        example = prompt + ann["output"]

        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        
        print(len(labels.tolist()))
        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
        }


def get_peft(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules = ["q_proj", "v_proj"],
        modules_to_save= ["embed_tokens", "lm_head"],

    )

    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, peft_config




def create_profiler(output_dir, enable_profiler=True):

    # Set up profiler
    if enable_profiler:
        wait, warmup, active, repeat = 1, 1, 2, 1
        total_steps = (wait + warmup + active) * (1 + repeat)
        schedule =  torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
        profiler = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{output_dir}/logs/tensorboard"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True)
        
        class ProfilerCallback(TrainerCallback):
            def __init__(self, profiler):
                self.profiler = profiler
                
            def on_step_end(self, *args, **kwargs):
                self.profiler.step()

        profiler_callback = ProfilerCallback(profiler)
    else:
        profiler = nullcontext()
        total_steps = nullcontext()
        profiler_callback = nullcontext()

    return profiler, total_steps, profiler_callback

def train(model, enable_profiler, profiler, profiler_callback, training_args, train_dataset, eval_dataset):


    with profiler:
        # Create Trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator,
            callbacks=[profiler_callback] if enable_profiler else [],
        )

        # Start training
        trainer.train()


def main(args):
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_dir)
    # Prepare dataset
    dataset_train = MusicDataset(args.data_dir, tokenizer, split="train")
    eval_dataset = MusicDataset(args.data_dir, tokenizer, split="validation")
    print(len(dataset_train[0]))

    # train_dataset = build_instruction_dataset(
    #                 data_path=args.data_dir,
    #                 tokenizer=tokenizer,
    #                 max_seq_length=2048,
    #                 data_cache_dir = None,
    #                 preprocessing_num_workers = 1)


    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=True,
        collate_fn=default_data_collator,
    )


    return  
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_dir)
    model =LlamaForCausalLM.from_pretrained(args.llama_model, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)
    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    if model_vocab_size != len(tokenizer):
        print(f"Resize model vocab size to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # load peft
    model, lora_config = get_peft(model)
    # resize model


    config = {
        'lora_config': lora_config,
        'learning_rate': 1e-4,
        'num_train_epochs': 1,
        'gradient_accumulation_steps': 2,
        'per_device_train_batch_size': 2,
        'gradient_checkpointing': False,
    }

    profiler, total_steps, profiler_callback = create_profiler(args.output_dir, args.enable_profiler)

    # Prepare dataset
    train_dataset = MusicDataset(args.data_dir, tokenizer, split="train")
    eval_dataset = MusicDataset(args.data_dir, tokenizer, split="validation")

    # Define training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        bf16=True,  # Use BF16 if available
        # logging strategies
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
        optim="adamw_torch_fused",
        max_steps=total_steps if args.enable_profiler else -1,
        **{k:v for k,v in config.items() if k != 'lora_config'}
    )

    
    results = train(model, args.enable_profiler, profiler, profiler_callback, training_args, train_dataset, eval_dataset)
    [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

    model.save_pretrained(args.output_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="../../dataset/processed_data", 
        help="Directory for the dataset"
    )
    parser.add_argument(
        "--tokenizer_dir", default="../../tokenizer/llama2music", 
        help=""
    )
    parser.add_argument(
        "--llama_model", default="/home/feiyuehchen/personality/llama/llama-2-7b-hf",
        help="Directory for the llama2 checkpoint"
    )
    parser.add_argument(
        "--enable_profiler", default=True,
        help="Define an optional profiler or not"
    )
    parser.add_argument(
        "--output_dir", default="tmp/llama-output",
        help="Define for the output"
    )





    args = parser.parse_args()

    main(args)

