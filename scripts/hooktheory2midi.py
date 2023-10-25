# modified from https://github.com/chrisdonahue/sheetsage/blob/main/notebooks/Dataset.ipynb

import json
import pretty_midi
import os
import numpy as np
from scipy.interpolate import interp1d
import math

def json2midi(save_dir, json_set):
    os.makedirs(save_dir, exist_ok = True)
    # Inspect one
    # data = train_set['kwxAaOrYxKG']
    # data = list(train_set.values())[0]
    
    
    # print(json.dumps(data, indent=2))

    for name, data in json_set.items():
        
        try:

            # Parse alignment


            beat_to_time_fn = interp1d(
                data['alignment']['refined']['beats'],
                data['alignment']['refined']['times'],
                kind='linear',
                fill_value='extrapolate')
            start_time = beat_to_time_fn(0)
            num_beats = data['annotations']['num_beats']
            end_time = beat_to_time_fn(num_beats)
            print(start_time, end_time)

            # Interpret annotation as MIDI

            CHORD_OCTAVE = 4
            MELODY_OCTAVE = 5



            midi = pretty_midi.PrettyMIDI()

            # Create click track
            click = pretty_midi.Instrument(program=0, is_drum=True)
            midi.instruments.append(click)
            beats_per_bar = data['annotations']['meters'][0]['beats_per_bar']
            for b in range(math.ceil(num_beats + 1)):
                downbeat = b % beats_per_bar == 0
                click.notes.append(pretty_midi.Note(
                    100 if downbeat else 75,
                    37 if downbeat else 31,
                    beat_to_time_fn(b),
                    beat_to_time_fn(b + 1)))

            # Create harmony track
            harmony = pretty_midi.Instrument(program=24)  # Acoustic Guitar (nylon)
            midi.instruments.append(harmony)
            for c in data['annotations']['harmony']:
                root_position_pitches = [c['root_pitch_class']]
                for interval in c['root_position_intervals']:
                    root_position_pitches.append(root_position_pitches[-1] + interval)
                for p in root_position_pitches:
                    harmony.notes.append(pretty_midi.Note(
                        67,
                        p + CHORD_OCTAVE * 12,
                        beat_to_time_fn(c['onset']),
                        beat_to_time_fn(c['offset'])
                    ))

            # Create melody track
            melody = pretty_midi.Instrument(program=0)
            midi.instruments.append(melody)
            for n in data['annotations']['melody']:
                melody.notes.append(pretty_midi.Note(
                    100,
                    n['pitch_class'] + (MELODY_OCTAVE + n['octave']) * 12,
                    beat_to_time_fn(n['onset']),
                    beat_to_time_fn(n['offset'])
                ))

            midi.write(os.path.join(save_dir, name+'.midi'))
            print(os.path.join(save_dir, name+'.midi'))
        except:
            print(name)


def main():
    #need to change by yourself
    with open('Hooktheory.json', 'r') as f:
        dataset = json.load(f)

    print(f"total: {len(dataset)}")

    # Filter dataset

    train_set = {
        k:v for k, v in dataset.items()
        if v['split'] == 'TRAIN'
        and 'AUDIO_AVAILABLE' in v['tags']
        and 'MELODY' in v['tags']
        and 'TEMPO_CHANGES' not in v['tags']}

    valid_set = {
        k:v for k, v in dataset.items()
        if v['split'] == 'VALID'
        and 'AUDIO_AVAILABLE' in v['tags']
        and 'MELODY' in v['tags']
        and 'TEMPO_CHANGES' not in v['tags']}
    test_set = {
        k:v for k, v in dataset.items()
        if v['split'] == 'TEST'
        and 'AUDIO_AVAILABLE' in v['tags']
        and 'MELODY' in v['tags']
        and 'TEMPO_CHANGES' not in v['tags']}

    print('================================')
    print('split dataset')
    print('================================')

    print(f"train: {len(train_set)}")
    print(f"valid: {len(valid_set)}")
    print(f"test: {len(test_set)}")

    # json2midi('./dataset/train', train_set)
    # json2midi('./dataset/valid', valid_set)
    # json2midi('./dataset/test', test_set)
    
    print('================================')
    print('json2midi')
    print('================================')
    print(f"train: {len(os.listdir('./dataset/train'))}")
    print(f"valid: {len(os.listdir('./dataset/valid'))}")
    print(f"test: {len(os.listdir('./dataset/test'))}")
    
    
if __name__ == "__main__":
    main()