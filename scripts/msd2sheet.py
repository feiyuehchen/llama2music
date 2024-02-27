#================================================================================================
# This file should be ran in the enviroment where Jukebox/Sheetstage is available.
# I created another conda enviroment for stability.
# Author: Fei Yueh Chen
#================================================================================================

import os
import glob
import json
import argparse

# parallel
from multiprocessing import Process, Pool
from itertools import repeat
import time

# Create MIDI
from io import BytesIO
import pretty_midi

def music2sheet(args_list):
    mp3_file, args = args_list
    

    midi_path = os.path.join(args.save_dir, mp3_file.replace('.mp3', '.mid'))
    if os.path.exists(midi_path):
        print(f"Midi file already exist: {mp3_file}")
        # return midi_path
        return
    else:
        print(mp3_file)
    USE_JUKEBOX = False
    # SEGMENT_START_HINT = 69
    # SEGMENT_END_HINT = 88
    # BPM_HINT = 76
    mp3_path = os.path.join(args.MSD_dir, mp3_file)
    
    lead_sheet, segment_beats, segment_beats_times = sheetsage(
        mp3_path,
        use_jukebox=USE_JUKEBOX)
        # segment_start_hint=SEGMENT_START_HINT,
        # segment_end_hint=SEGMENT_END_HINT,
        # beats_per_minute_hint=BPM_HINT)
    beat_to_time_fn = create_beat_to_time_fn(segment_beats, segment_beats_times)
    midi_bytes = lead_sheet.as_midi(beat_to_time_fn)
    midi = pretty_midi.PrettyMIDI(BytesIO(midi_bytes))
    midi.write(midi_path)
    
    # return midi_path


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
    
def main(args):
    f = open(args.MSD_dir.replace('mp3', 'unique_tracks.txt'), 'r')
    mp3_list = []
    for line in f.readlines():
        mp3_file = line.split('<SEP>')[0]+'.mp3'
        mp3_list.append(mp3_file)
    
    if args.slice_start:
        mp3_list = mp3_list[args.slice_start:args.slice_end]
    

    if args.start_previos:
        previos_finished = len(os.listdir(args.save_dir))
        print(f'========================== START FROM {previos_finished} ========================')
        mp3_list = mp3_list[previos_finished:]
    
    
    start_time = time.time()
    
    for subset in batch(mp3_list, args.batch):
        substart_time = time.time()
        args_list = [(mp3_file, args) for mp3_file in subset]
        with Pool(args.multiprocess) as p:
            p.map(music2sheet, args_list)

        
        print(f"batch total time: {time.time()-substart_time}")
        

    print(f"all total time: {time.time()-start_time}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--MSD_dir", default='../../music_dataset/MSD/mp3', 
        help="Directory for the mp3 files of Million Song Dataset.",
    )
    
    parser.add_argument(
        "--sheetstage_dir", default='../../music_dataset/sheetstage', 
        help="Directory for the sheetstage. Only for import.",
    )
    parser.add_argument(
        "--save_dir", default='../../music_dataset/msd_sheet/mid', 
        help="Directory for the sheetstage. Only for import.",
    )
    parser.add_argument(
        "--multiprocess", default= 32, type=int,
        help="parallel run",
    )
    parser.add_argument(
        "--batch", default= 1000, type=int, 
        help="batch for every saving epoch",
    )
    parser.add_argument(
        "--slice_start", default=None, type=int,
        help="slice the mp3_list"
    )
    parser.add_argument(
        "--slice_end", default=None, type=int,
        help="slice the mp3_list"
    )
    parser.add_argument(
        "--start_previos", default=False,
        help="start from the previos or not"
    )
    
    

    
    args = parser.parse_args()
    import sys
    sys.path.append(args.sheetstage_dir)

    from sheetsage.infer import sheetsage
    from sheetsage.align import create_beat_to_time_fn
    
    os.makedirs(args.save_dir, exist_ok = True)
    main(args)
    
    