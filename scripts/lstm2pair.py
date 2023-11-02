from gensim.models import Word2Vec
import os
import numpy as np
import pandas as pd
import pickle
import argparse


def test(args):
    seqlength = 20
    num_midi_features = 3
    num_sequences_per_song = 2
    training_rate = 0.8
    validation_rate = 0.1
    test_rate = 0.1
    

    syll_model_path = os.path.join(args.data_dir, 'enc_models/syllEncoding_20190419.bin') 
    word_model_path = os.path.join(args.data_dir, 'enc_models/wordLevelEncoder_20190419.bin') 

    songs_path = os.path.join(args.data_dir,  'data/songs_word_level')

    print('Creating a dataset with sequences of length', seqlength, 
        'with', num_sequences_per_song, 'sequences per song')

    syllModel = Word2Vec.load(syll_model_path)
    wordModel = Word2Vec.load(word_model_path)
    syll2Vec = syllModel.wv['Hello']
    word2Vec = wordModel.wv['world']
    num_syll_features = len(syll2Vec) + len(word2Vec)
    print('Syllable embedding length :', num_syll_features)
    files = os.listdir(songs_path)
    num_songs = len(files)
    print("Total number of songs : ", num_songs)
    
    seq_filename_list = [] # to keep track of filename from which a sequence is extracted
    small_file_cntr = 0 # to keep track of files with less than 20 syllable-note pairs
    
    sample_file = files[0]
    sample_features = np.load(os.path.join(songs_path, sample_file), allow_pickle=True) # load midi files to feature
    print(len(sample_features), len(sample_features[0]))

    # import midi
    import sys
    sys.path.append('/home/feiyuehchen/personality/python-midi')
    import src.fileio as midi
    
    def save_midi_pattern(filename, midi_pattern):
        if filename is not None:
            midi.write_midifile(filename, midi_pattern)



    save_midi_pattern('temp.mid', sample_features)
    

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default='../../music_dataset/LSTM_GAN', 
        help="path for the directory of LSTM-GAN",
    )
    parser.add_argument(
        "--lyrics_path", default='../dataset/lyrics/raw/', 
        help="path for the directory of LSTM-GAN",
    )

    
    args = parser.parse_args()
    test(args)