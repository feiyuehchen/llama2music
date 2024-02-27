from transformers import LlamaTokenizer
import os
from datasets import load_dataset, load_from_disk
import argparse
import numpy as np
from tqdm import tqdm
import json
import pandas as pd


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
def get_dataset_length(data_path, tokenizer):
    all_dataset = load_from_disk(data_path)
    df = pd.DataFrame(columns = ['split', 'dataset_type', 'length'])

    for split in ['train', 'validation', 'test']:
        dataset = all_dataset[split]
        for data in tqdm(dataset):            
            if data.get("input", "") == "":
                prompt = PROMPT_DICT["prompt_no_input"].format_map(data)
            else:
                prompt = PROMPT_DICT["prompt_input"].format_map(data)
            example = prompt + data["output"]
            # +1 is for the eos token
            df.loc[len(df)] = [split, data["dataset_type"], len(tokenizer.tokenize(example))+1]

    return df




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="../../dataset/processed_data", 
        help="path for the json data, you could run token2script.py to get it"
    )
    parser.add_argument(
        "--tokenizer_dir", default="../../tokenizer/planB", 
        help="path for the tokenizer"
    )
    parser.add_argument(
        '--save_path', default='./result/length/planB.csv', type=str
    )
    args = parser.parse_args()

    # load dataset
    print("="*60)
    print(f"load tokenizer from {args.tokenizer_dir}")
    print(f"load dataset from {args.data_dir}")
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_dir)
    df = get_dataset_length(args.data_dir, tokenizer)

    print(f"mean: {np.mean(df['length'])}")
    print(f"std: {np.std(df['length'])}")
    print(f"max: {max(df['length'])}")

    print("="*60)


    # save results
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    df.to_csv(args.save_path)
