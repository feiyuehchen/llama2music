import os
import csv
import json
from tqdm import tqdm
import pandas as pd

def read_metro_csv(metro_raw_path):
    # metro_column_name = ['index', 'song', 'year', 'artist', 'genre', 'lyrics']
    
    metro_csv = pd.read_csv(metro_raw_path)
    metro_csv = metro_csv.drop(columns = 'index')
    
    print(f"total lyrics before cleaning: {len(metro_csv)}")
      
    clean_metro_csv = []
    for id in tqdm(range(len(metro_csv))):
        clean_metro_csv.append([str(metro_csv['song'][id]).replace('-', ' '), # song
                                str(metro_csv['artist'][id]).replace('-', ' '), # artist
                                metro_csv['year'][id], # year
                                metro_csv['lyrics'][id]  # lyrics
                                ])
    
    print(f"total lyrics after cleaning: {len(clean_metro_csv)}")

    return clean_metro_csv

def csv2json(metro_csv):
    metro_json = []
    # metro_column_name = ['ARTIST_NAME', 'ARTIST_URL', 'SONG_NAME', 'SONG_URL', 'LYRICS']
    for row in tqdm(metro_csv):
        # add lyrics information in input
        metro_json.append(
            {
                "instruction": "",
                "input": f"""Song: {row[0]}
Songwriter: {row[1]}
Year: {row[2]}
Lyrics:
{row[3]}""",
                "output": ""
            }
        )
    
    print(metro_json[0])
    return metro_json


if __name__ == "__main__":
    # you may change the directory
    metro_raw_path = '../../music_dataset/az_lmd_dataset/metrolyrics.csv'
    metro_json_save_path = '../dataset/lyrics/metro.json'
    
    metro_csv = read_metro_csv(metro_raw_path)
    metro_json = csv2json(metro_csv)
    
    with open(metro_json_save_path, 'w') as f:
        json.dump(metro_json, f)
    