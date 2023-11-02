import openai
import yaml
import argparse
import os
import json
from tqdm import tqdm
from langdetect import detect
import random
import signal
import time


# parallel
from multiprocessing import Process, Pool



# OpenAI creds
def getcreds(cred_path):
    with open(cred_path, 'r') as file:
        # Load the YAML data
        creds = yaml.safe_load(file)
        openai.api_key = creds["OPENAI_API_KEY"]
        openai.organization = creds["OPENAI_ORG_ID"]
    
    
def load_dataset(args):
    if args.clean_path:
        print('load clean dataset ...')
        f = open(args.clean_path)
        clean_lyrics_dataset = json.load(f)
        print(f"length of the clean dataset: {len(clean_lyrics_dataset)}")    


    else:
        print('combining dataset ...')
        print('============================')
        lyrics_dataset = []
        for json_name in os.listdir(args.lyrics_dir):
            print(json_name)
            f = open(os.path.join(args.lyrics_dir, json_name))
            temp_data = json.load(f)
            lyrics_dataset += temp_data
        
        print(f"dataset with all languages: {len(lyrics_dataset)}")
        
        clean_lyrics_dataset = []
        for lyrics in tqdm(lyrics_dataset):
            if detect(lyrics) == 'en':
                clean_lyrics_dataset.append(lyrics)
        
        print(f"dataset with English only: {len(clean_lyrics_dataset)}")    
        
        with open(args.clean_lyrics_save_path, 'w') as f:
            json.dump(clean_lyrics_dataset, f)
            
    
    # give id
    id_clean_lyrics_dataset = []
    clean_lyrics_dataset = sorted(clean_lyrics_dataset)
    for id in range(len(clean_lyrics_dataset)):
        id_clean_lyrics_dataset.append({
            'id' : id,
            'lyrics': clean_lyrics_dataset[id]
        })
    
    return id_clean_lyrics_dataset



def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lyrics_dir", default='../dataset/lyrics/raw', 
        help="directory for lyrics datasets",
    )
    parser.add_argument(
        "--cred_path", default='../assets/credentials.yaml', 
        help="path for the openAI credentials",
    )
    parser.add_argument(
        "--save_path", default='../dataset/lyrics2text/lyrics2text.json', 
        help="save path for the output lyrics and text",
    )
    parser.add_argument(
        "--clean_lyrics_save_path", default='../dataset/lyrics/clean_lyrics.json', 
        help="save path for the clean lyrics dataset",
    )
    parser.add_argument(
        "--clean_path", default= None, 
        help="if there is a clean lyrics dataset already",
    )
    parser.add_argument(
        "--previos_json", default= None, 
        help="if there is lyrics2.json and you want to append it",
    )
    parser.add_argument(
        "--multiprocess", default= 48, 
        help="parallel run GPT completion",
    )
    parser.add_argument(
        "--batch", default= 10000, 
        help="batch for every saving epoch",
    )

    

    
    
    args = parser.parse_args()
    getcreds(args.cred_path)
    lyrics_dataset = load_dataset(args)
    random.shuffle(lyrics_dataset)
    

    instruction = """Please answer the following questions, with each question being answered around 100 words:
1. Analyze the lyrics and determine the overall sentiment (e.g., happy, sad, angry) of the song. Provide a summary of the emotional tone.
2. Identify the tone (e.g., humorous, melancholic, rebellious) and mood (e.g., relaxed, upbeat, reflective) of the song.
3. Pick out and summarize the most iconic or memorable lines from the song.
4. Please analyze the lyrics and classify them into one of the four categories based on Russell's Circumplex model of affect: (1) High Positive Affect, (2) High Negative Affect, (3) Low Positive Affect, or (4) Low Negative Affect. """
    response_dataset = []
    id_set = set()
    if args.previos_json:
        print(f'load previos json file: {args.previos_json}')
        f = open(args.previos_json)
        response_dataset = json.load(f)
        for item in response_dataset:
            id_set.add(item['lyrics_id'])
    
    print(f'id_set: {len(id_set)}')    
    print(f'response_dataset: {len(response_dataset)}')
    def run_GPT_map(item):
        

        # for item in tqdm(lyrics_dataset):

        time.sleep(random.uniform(1.0,3.0))
        def handler(signum, frame):
            print("openai api timeout!")
            # pass
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(20)
        
        try:
            # skip the lyrics which is analyzed before
            if item['id'] in id_set:
                # continue
                # print('analyzed before:', item['id'])
                return
            else:
                id_set.add(item['id'])

            
            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}

### Input:
""" 

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt+item['lyrics']}]
            )   
        
            # print(response['choices'][0]['message']['content'])
            # print(response)
            # response_dataset.append({
            #     'lyrics_id': item['id'],
            #     'instuction': instruction,
            #     'input': item['lyrics'],
            #     'output': response['choices'][0]['message']['content']
            #     })
            

            # # save
            # if len(response_dataset) == 10 or len(response_dataset)%10000 == 0:
            #     save_path = args.save_path.replace('.json', f'_{len(response_dataset)}.json')
            #     with open(save_path, 'w') as f:
            #         json.dump(response_dataset, f)

            
            # print(f"length of the response dataset: {len(response_dataset)}")
            bind = {
                    'lyrics_id': item['id'],
                    'instuction': instruction,
                    'input': item['lyrics'],
                    'output': response['choices'][0]['message']['content']
                    }
            # print("success:", item['id'])

        except Exception: 
            bind = None
        
        signal.alarm(0)
        
        return bind
        
    count = 0
    for subset in batch(lyrics_dataset, args.batch):
        print('======================================================')

        print(f"batch count: {count}")
        count += 1
        with Pool(args.multiprocess) as p:
            progress_bar = tqdm(total=len(subset))
            response_dataset += tqdm(p.imap(run_GPT_map, subset), total=len(subset))
        print(f"{len(response_dataset)}/{len(lyrics_dataset)}")
        print('remove NULL')
        response_dataset = [i for i in response_dataset if i is not None]
        print(f"{len(response_dataset)}/{len(lyrics_dataset)}")
        save_path = args.save_path.replace('.json', f'_{len(response_dataset)}.json')
        with open(save_path, 'w') as f:
            json.dump(response_dataset, f)





    print('======================================================')
    print(f"{len(response_dataset)}/{len(lyrics_dataset)}")
    print('remove NULL')
    response_dataset = [i for i in response_dataset if i is not None]
    print(f"{len(response_dataset)}/{len(lyrics_dataset)}")

    with open(args.save_path, 'w') as f:
        json.dump(response_dataset, f)
    
    