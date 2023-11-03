import os
import csv
import json
from tqdm import tqdm


def read_az_csv(az_root_dir):
    az_csv = []
    # az_column_name = ['ARTIST_NAME', 'ARTIST_URL', 'SONG_NAME', 'SONG_URL', 'LYRICS', ...]
    for csv_file_name in os.listdir(az_root_dir):
        with open(os.path.join(az_root_dir, csv_file_name), 'r') as csv_file:
            temp = [row for row in csv.reader(csv_file) if row]
            az_csv += temp[1:]
    
    print(f"total lyrics before cleaning: {len(az_csv)}")
    
    clean_az_csv = []
    for row in az_csv:
        if len(row)==5 and row[4]:
            clean_az_csv.append([row[0], # artist
                                 row[1], # artitst url
                                 row[2], # song name
                                 row[3], # song url
                                 row[4]  # lyrics
                                 ])
    
    print(f"total lyrics after cleaning: {len(clean_az_csv)}")

    return clean_az_csv

def csv2json(az_csv):
    az_json = []
    # az_column_name = ['ARTIST_NAME', 'ARTIST_URL', 'SONG_NAME', 'SONG_URL', 'LYRICS']
    for row in tqdm(az_csv):
        # add lyrics information in input
#         az_json.append(
#             {
#                 "instruction": "",
#                 "input": f"""Song: {row[2]}
# Songwriter: {row[0]}
# Lyrics:
# {row[4]}""",
#                 "output": ""
#             }
#         )
        az_json.append(f"""Song: {row[2]}
Songwriter: {row[0]}
Lyrics:
{row[4]}""")
    
    print(az_json[0])
    
    return az_json

if __name__ == "__main__":

    # you may change the directory
    az_root_dir = '../../music_dataset/azlyrics-scraper'
    az_json_save_dir = '../dataset/lyrics/raw'
    
    az_csv = read_az_csv(az_root_dir)
    az_json = csv2json(az_csv)
    os.makedirs(az_json_save_dir, exist_ok = True)
    az_json_save_path = os.path.join(az_json_save_dir, 'az.json')
    
    with open(az_json_save_path, 'w') as f:
        json.dump(az_json, f)
    
    
    

    
    
    