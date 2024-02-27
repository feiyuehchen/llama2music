# input format:
# lyrics2text data:
# [
#     {
#         "id": name,
#         "instuction": "Please answer the following questions, with each question being answered around 100 words:\n1. Analyze the lyrics and determine the overall sentiment (e.g., happy, sad, angry) of the song. Provide a summary of the emotional tone.\n2. Identify the tone (e.g., humorous, melancholic, rebellious) and mood (e.g., relaxed, upbeat, reflective) of the song.\n3. Pick out and summarize the most iconic or memorable lines from the song.\n4. Please analyze the lyrics and classify them into one of the four categories based on Russell's Circumplex model of affect: (1) High Positive Affect, (2) High Negative Affect, (3) Low Positive Affect, or (4) Low Negative Affect. ",
#         "input": "Lyrics: ...",
#         "output": ...
#     },
#     {
#         ...
#     }
# ]
# output format
# [
#     {
#         "id" : name,
#         "tokens": "Lyrics: ...",
#         "instruction": ...,
#         "control_code": ...
#     }
# ]


import os
import argparse
import json
from tqdm import tqdm
import random

from transformers import pipeline


RUSSEL_TYPE = ["Low Positive Affect", 
               "Low Negative Affect", 
               "High Positive Affect", 
               "High Negative Affect"]
    
def lyrics2token(args):

    save_path = f"{args.save_dir}/{args.save_name}"
    
    # read lyrics and text
    with open(args.data_path, 'r') as f:
        lyrics2text_data = json.load(f)
    
    random.seed(20)
    random.shuffle(lyrics2text_data)    
        
    output_json = []
    # if args.russel_balance:
    russel = {}
    for affect in RUSSEL_TYPE:
        russel[affect] = 0
    
    if args.filter_toxic:
        toxigen_roberta = pipeline("text-classification", model="tomh/toxigen_roberta")


    now = 0
    for data in tqdm(lyrics2text_data):
        if args.filter_toxic:
            if toxigen_roberta(data["input"][:512])[0]['label'] == "LABEL_1":
                continue

        try:
            one, left = data["output"].replace('\n', '').replace('1.', '').split('2.')
            two, left = left.split('3.')
            three, four = left.split('4.')
            condition_data = [one, two, three, four]
        except:
            continue

        
        new_data = {
            "id": data["id"],
            "tokens": data["input"].strip("Lyrics:"),
            "control_code": condition_data[now%4].strip(' '),
            "instruction": "Please provide me with lyrics based on the following text.",
            "russel_info": None
        }
        
        # russel
        if now%4 == 3:
            for affect in RUSSEL_TYPE:
                if affect in condition_data[now%4]:
                    russel[affect] += 1
                    new_data["russel_info"] = affect
        output_json.append(new_data)
        
        if len(output_json) == args.slice:
            print(f'Reach the slice: {args.slice}')
            break
        
        now += 1  
    print('russel')
    print(russel) 
    # save output
    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    if args.split_to_train_valid:
        valid_set, train_set  = output_json[:int(len(output_json)*args.valid_ratio)], output_json[int(len(output_json)*args.valid_ratio):]
        print(f"Number of files in train set: {len(train_set)}")
        print(f"Number of files in valiation set: {len(valid_set)}")
        train_save_path = save_path.replace('.json', '_train.json')
        with open(train_save_path, 'w') as f:
            json.dump(train_set, f, indent = 4)
        valid_save_path = save_path.replace('.json', '_validation.json')
        with open(valid_save_path, 'w') as f:
            json.dump(valid_set, f, indent = 4)
    else:
        with open(save_path, 'w') as f:
            json.dump(output_json, f, indent = 4)
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", 
        help="directory of lyrics and text files in json format",
    )
    parser.add_argument(
        "--save_dir", 
        help="save directory for the output json file"
    )
    parser.add_argument(
        "--save_name", default='lyrics.json',
        help="save name for the output json file"
    )
    parser.add_argument(
        "--slice", default=None, type=int,
        help="if you want to slice the dataset, set a integer." 
    )
    parser.add_argument(
        "--split_to_train_valid", default=True,
        help="split the dataset into train and validation"
    )
    parser.add_argument(
        "--valid_ratio", default=0.05, type=int,
        help="ratio for the validation/total dataset"
    )
    parser.add_argument(
        "--filter_toxic", default=True,
        help="use toxigen roberta or not"
    )


    lyrics2token(args)



