# Turn token files to Stanford Alpaca format.
# midi token format:
# [
#     {
#     "midiname": ...,
#     "dataset_type": ...,
#     "token_type": ...,
#     "token_length": ...,
#     "tokens": ...,
#     "control_code": ... },
# ]

# Stanford Alpaca format:
# [
#   {"instruction" : ...,
#    "input" : ...,
#    "output" : ...
#    "dataset_type": ..., # identify datasets
#   }, 
#   ...
# ]


import argparse
import os
import json
from tqdm import tqdm
import random

from datasets import load_dataset


def tokens2script(token_dir, temp_dir, min_tok_len, set_type):
    subset_dir = os.listdir(f"{token_dir}/{set_type}")
    print(subset_dir)

    
    for json_file in subset_dir:
        # load token json data under token directory
        with open(f"{token_dir}/{set_type}/{json_file}") as f:
            token_data = json.load(f)
        print(f"Name: {json_file}")
        print(f"Number of data: {len(token_data)}")
        output_json = []

        # turn tokens into stanford alpaca format
        for id, data in tqdm(enumerate(token_data)):
            if "token_length" in data:
                if data["token_length"] < min_tok_len:
                    continue
            try:
                # Stanford Alpaca format     
                new_data = {
                    "instruction" : data["instruction"],
                    "input" : data["control_code"],
                    "output" : data["tokens"],
                    "dataset_type": data["dataset_type"]
                }
                output_json.append(new_data)
            except:
                pass

        # print(f"Example Data: {output_json[0]}")
        print(f"Length of totoal json files: {len(output_json)}")
        # save tmp json
        os.makedirs(f"{temp_dir}/{set_type}", exist_ok=True)
        save_path = f"{temp_dir}/{set_type}/{json_file}"
        print(f"Save as: {save_path}")
        with open(save_path, 'w') as f:
            json.dump(output_json, f)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir", default="../../dataset/ailab17k_data/seg_processed_data", 
        help="save path for the output json file(train and validation)"
    )
    parser.add_argument(
        "--token_dir", default="../../dataset/ailab17k_data/seg_json_data", 
        help="directory for tokens.",
    )
    parser.add_argument(
        "--temp_dir", default="../../dataset/ailab17k_data/seg_tmp", 
        help="temp folder",
    )
    # parser.add_argument(
    #     "--max_tok_len", default = 12000,
    #     help='max token length, use for filter'
    # )
    parser.add_argument(
        "--min_tok_len", default = 64,
        help='min token length, use for filter'
    )

    args = parser.parse_args()

    data_files = dict()

    for set_type in ["test", "validation", "train"]:
        tokens2script(args.token_dir, args.temp_dir, args.min_tok_len, set_type)
        subset_dir = [f"{args.temp_dir}/{set_type}/{json_file}" for json_file in os.listdir(f"{args.token_dir}/{set_type}")]
        data_files[set_type] = subset_dir
        
    print(data_files)
    dataset = load_dataset('json', data_files=data_files)

    dataset.save_to_disk(args.save_dir)
    # dataset.save_to_disk(f"{args.save_dir}/test.json")
