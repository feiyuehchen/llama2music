import os
from pydub import AudioSegment
from tqdm import tqdm
import argparse

def traverse_dir(root_dir,
                extension=('mid', 'MID', 'midi'),
                is_sort = False, 
                is_pure = False
                ):
    file_list = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path
                file_list.append(pure_path)
    if is_sort:
        print("sort the file list")
        file_list.sort()
    print(f"total count of the file: {len(file_list)}")
    return file_list


def midi2wav(args):
    file_list = traverse_dir(args.midi_dir, is_pure=True, is_sort=True)
    
    for midi_file in tqdm(file_list):
        
        wav_name = midi_file.replace(os.path.splitext(midi_file)[1], '.wav')
        wav_file = os.path.join(args.wav_dir, wav_name)
        midi_file = os.path.join(args.midi_dir, midi_file)
        print(wav_file)
        if not os.path.exists(wav_file):
            # save
            fn = os.path.basename(wav_file)
            os.makedirs(wav_file[:-len(fn)], exist_ok=True)
            os.system(f'fluidsynth -ni {args.soundfont} {midi_file} -F {wav_file} -r {args.audio_frame_rate}')

    
def wav2mp3(args):
    file_list = traverse_dir(args.wav_dir, extension='wav', is_pure=True, is_sort=True)
    
    for wav_file in tqdm(file_list):
        mp3_file = os.path.join(args.mp3_dir, wav_file.replace('.wav', '.mp3'))
        fn = os.path.basename(mp3_file)
        os.makedirs(mp3_file[:-len(fn)], exist_ok=True)
        wav_file = os.path.join(args.wav_dir, wav_file)

        audio = AudioSegment.from_file(wav_file)
        audio = audio.set_frame_rate(args.audio_frame_rate)
        audio.export(mp3_file, format='mp3')


    
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
        "--extention", default=('mid', 'MID', 'midi'), 
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

    os.makedirs(args.wav_dir, exist_ok=True)
    os.makedirs(args.mp3_dir, exist_ok=True)
    
    midi2wav(args)
    wav2mp3(args)
    
