import os
from pydub import AudioSegment
from tqdm import tqdm
import numpy as np
import argparse




def midi2wav(args):
    soundfont = args.soundfont
    # you may specify the root
    root = '../irishman/wav'
    midi_train_path = '../irishman/midi/train'
    midi_valid_path = '../irishman/midi/validation'
    wav_train_path = os.path.join(root, 'train')
    wav_valid_path = os.path.join(root, 'validation')
    os.makedirs(wav_train_path, exist_ok=True)
    os.makedirs(wav_valid_path, exist_ok=True)
    print('train set')
    
    midi_path = midi_train_path
    wav_path = wav_train_path
    import random
    midi_list = os.listdir(midi_path)
    random.shuffle(midi_list)
    
    for midi_filename in tqdm(midi_list):
        midi_file = os.path.join(midi_path, midi_filename)
        wav_file = os.path.join(wav_path, midi_filename.replace('mid', 'wav'))
        if not os.path.exists(wav_file):
            os.system(f'fluidsynth -ni {soundfont} {midi_file} -F {wav_file} -r 44100')
    
    
def wav2mp3(args):
    root = '../irishman/wav'
    wav_train_path = os.path.join(root, 'train')
    wav_valid_path = os.path.join(root, 'validation')
    
    def turnmp3(path, setname = None):
        if setname=='train_set':
            wav_list = os.listdir(path)
            wav_list = np.array_split(wav_list, 20)
            for id, wav_sublist in enumerate(wav_list):
                
                wav_sub_path = os.path.join(path.replace('wav', 'mp3'), str(id))
                print(wav_sub_path)
                os.makedirs(wav_sub_path, exist_ok=True)
                for wav_filename in tqdm(wav_sublist):
                    if not wav_filename.endswith('.wav'):
                        continue
                    wav_file = os.path.join(path, wav_filename)
                    mp3_file = os.path.join(wav_sub_path, wav_filename).replace('wav', 'mp3')
                    
                    # print(wav_file)
                    # print(mp3_file)
                    audio = AudioSegment.from_file(wav_file)
                    audio = audio.set_frame_rate(44100)
                    audio.export(mp3_file, format='mp3')
        else:                
            for wav_filename in tqdm(path):
                if not wav_filename.endswith('.wav'):
                    continue
                wav_file = os.path.join(path, wav_filename)
                mp3_file = wav_file.replace('wav', 'mp3')
                
                # print(wav_file)
                # print(mp3_file)
                audio = AudioSegment.from_file(wav_file)
                audio = audio.set_frame_rate(44100)
                audio.export(mp3_file, format='mp3')

        
        
        
    # turnmp3(wav_valid_path)
    turnmp3(wav_train_path, 'train_set')

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--midi_dir", required=True, type=str,
        help="path to the midi data",
    )
    parser.add_argument(
        "--wav_dir", required=True, type=str,
        help="soundfont to convert midi to wav",
    )
    parser.add_argument(
        "--mp3_dir", required=True, type=str,
        help="soundfont to convert midi to wav",
    )
    parser.add_argument(
        "--extention", default=['midi', 'mid'], type=list,
        help="soundfont to convert midi to wav",
    )
    parser.add_argument(
        "--soundfont", default='../assets/GeneralUser_GS.sf2', type=str,
        help="soundfont to convert midi to wav",
    )
    parser.add_argument(
        "--audio_frame_rate", default=44100, type=int,
        help="soundfont to convert midi to wav",
    )
    
    
    args = parser.parse_args()
    midi2wav(args)
    wav2mp3(args)
    
