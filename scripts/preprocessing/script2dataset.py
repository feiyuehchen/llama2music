from transformers import pipeline

import argparse
import os
import json
import yaml
from tqdm import tqdm


def combine_dataset(args):
    with open(args.data_path, 'r') as f:
        data = yaml.load(f)
    output_json = []
    output_json.append({
        'Train': data[args.data_key_name]['Train'],
        'Validation': data[args.data_key_name]['Validation']
    }
    )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path", 
        help="save path for the output json file"
    )
    parser.add_argument(
        "--data_path", default="dataset.yaml",
        help="a yaml file includes json paths which need to combine",
    )
    parser.add_argument(
        "data_key_name",
        help="key name in data_path"
    )
    parser.add_argument(
        "--dataset", default=None, choices=[None, 'Irishman'],
        help="if the dataset has unique procedure"
    )

    args = parser.parse_args()
    
    combine_dataset(args)