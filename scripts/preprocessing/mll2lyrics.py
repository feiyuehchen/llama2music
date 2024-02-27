
import os
import numpy as np
import pandas as pd
import pickle
import argparse
import pandas as pd
from langdetect import detect
import json
from tqdm import tqdm


def main(args):
    
    csv_list = ['train', 'test']
    
    for dataset in csv_list:
        
        csv_path = os.path.join(args.data_dir, dataset+'.csv')
    
        df = pd.read_csv(csv_path)
        print(len(df))
        print(df[:5])

        
        mll_json = []
        # train.csv
        
        # column name: 'Artist', 'Song', 'Genre', 'Language', 'Lyrics'

        if dataset == 'train':
            print('====================================TRAIN SET START====================================')
            # get English only
            print(f'all language: {len(df)}')
            df = df[df['Language'] == 'en']
            print(f'English only: {len(df)}')
            for _, row in tqdm(df.iterrows()):
                mll_json.append([f"""Lyrics:{row[4]}""",  
f"""Song: {row[1]}
Songwriter: {row[0]}
Genre: {row[2]}"""])
            
            print('====================================TRAIN SET END====================================')
        # test.csv
        # column name: 'Song', 'Song year', 'Artist', 'Genre', 'Lyrics', 'Track_id'
        elif dataset == 'test':
            print('====================================TEST SET START====================================')

            for _, row in tqdm(df.iterrows()):
                mll_json.append([f"""Lyrics: {row[4]}""", 
f"""Song: {row[0]}
Song year: {row[1]}
Songwriter: {row[2]}
Genre: {row[3]}"""])
                
            print('====================================TEST SET END====================================')
            
        # print(mll_json[0])
        print(len(mll_json))
        with open(os.path.join(args.lyrics_dir, 'mml_'+dataset+'.json'), 'w') as f:
            json.dump(mll_json, f, indent=4)
    


        
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        help="path for the directory of Multi-Lingual Lyrics for Genre Classification(MLL)",
    )
    parser.add_argument(
        "--lyrics_dir", 
        help="directory for saving lyrics files",
    )


    args = parser.parse_args()
    os.makedirs(args.lyrics_dir, exist_ok = True)

    main(args)