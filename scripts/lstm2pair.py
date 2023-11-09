from gensim.models import Word2Vec
import os
import numpy as np
import pandas as pd
import pickle
import argparse
from miditoolkit.midi import parser as mid_parser  
from miditoolkit.midi import containers as ct
import json
from tqdm import tqdm



def main(args):
    seqlength = 20
    num_midi_features = 3
    velocity = 60
    
    songs_path = os.path.join(args.data_dir,  'data/songs_word_level')


    files = os.listdir(songs_path)
    print(files[:5])
    num_songs = len(files)
    print("Total number of songs : ", num_songs)
    
    small_file_cntr = 0 # to keep track of files with less than 20 syllable-note pairs
    
    # sample_file = files[0]
    # sample_features = np.load(os.path.join(songs_path, sample_file), allow_pickle=True) # load midi files to feature
    # print(len(sample_features), len(sample_features[0]))   

    #     if filename is not None:
    #         midi.write_midifile(filename, midi_pattern)


    # destination = "test.mid"
    # midi_pattern.write(destination)
    


    
    # load all the songs, cut to 20 note-sequence, convert to song embeddinds

    lstm_json = []
    

    for file in tqdm(files):
        features = np.load(os.path.join(songs_path, file), allow_pickle=True) # load midi files to feature
        if len(features[0][1]) >= seqlength: # seqlength = 20, if length of song > 20 note
            word = ''
            # create an empty file
            midi_obj = mid_parser.MidiFile()
            beat_resol = midi_obj.ticks_per_beat # 480
            # create an  instrument
            track = ct.Instrument(program=0, is_drum=False, name='melody')
            midi_obj.instruments = [track]
            prev_end = 0
            
            # turn features[0][3] from [['but'], ['o', 'ver']] to ['but', 'o', 'ver']
            new_syllList = []
            for syllList in features[0][3]:
                for syll in syllList:
                    new_syllList.append(syll)

            # take midi and syllable
            for midiList, syllList in zip(features[0][1], features[0][3]):
                
                for midi, syll in zip(midiList, syllList):
                    # for each midi in midiList:
                    # [[76.0, 0.25, 0.0]]
                    # [[pitch, duration, rest]] 
                    # create one note      
                    
                    pitch = int(midi[0])
                    duration = int(beat_resol*midi[1])
                    rest = int(beat_resol*midi[2])
                    start = prev_end
                    end = prev_end + duration
                    note = ct.Note(start=start, end=end, pitch=pitch, velocity=velocity)
                    midi_obj.instruments[0].notes.append(note)
                    # create one marker(syllable)
                    # [['but'], ['o', 'ver']]
                    word += syll
                    marker = ct.Marker(time=start, text=syll)
                    midi_obj.markers.append(marker)
                    # prepare_next
                    prev_end = end+rest
                    
                # if there is a rest, add a newline
                if rest > 0: 
                    word += '\n'
                    marker = ct.Marker(time=start, text='\n')
                    midi_obj.markers.append(marker)
                else:
                    word += ' '
                
                
                
            
                
            
            # print(word)
            # print(midi_obj.markers)
            # for note in midi_obj.instruments[0].notes:
            #     print(note)
            
            
            # write to file
            midi_obj.dump(os.path.join(args.midi_dir, file.replace('npy', 'mid')))
            lstm_json.append(f"""Lyrics:
{word}"""    
        )
        
        else: # seqlength < 20
            small_file_cntr += 1
    
    with open(os.path.join(args.lyrics_dir, 'lstm.json'), 'w') as f:
        json.dump(lstm_json, f)
    
       
def split_dataset(args):
    f = open(os.path.join(args.lyrics_dir, 'lstm.json'))
    all_dataset = json.load(f)
    
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default='../../music_dataset/LSTM_GAN', 
        help="path for the directory of LSTM-GAN",
    )
    parser.add_argument(
        "--midi_dir", default='../../music_dataset/llama2music/dataset/music/raw/LSTM_GAN', 
        help="directory for saving midi files",
    )
    parser.add_argument(
        "--lyrics_dir", default='../../music_dataset/llama2music/dataset/lyrics/raw', 
        help="directory for saving lyrics files",
    )
    parser.add_argument(
        "--split_ratio", default=0.01, 
        help="directory for saving lyrics files",
    )    


    
    args = parser.parse_args()
    os.makedirs(args.midi_dir, exist_ok = True)
    os.makedirs(args.lyrics_dir, exist_ok = True)

    main(args)