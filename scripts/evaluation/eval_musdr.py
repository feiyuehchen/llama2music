import os, pickle
from remi.utils import (
    read_items,
    quantize_items,
    extract_chords,
    group_items,
    item2event
)
from fractions import Fraction
from musdr.utils import(
    compute_histogram_entropy,
    compute_piece_pitch_entropy,
    compute_piece_groove_similarity,
    compute_piece_chord_progression_irregularity,
    compute_structure_indicator
)
import pandas as pd
import argparse

midi_dir = "/home/feiyuehchen/personality/llama2music/generation/B_REMI_2048_32_128_epoch_3"
out_csv = "./B_REMI_2048_32_128_epoch_3.csv"



CHORD_DICT = {'C:':1, 'C#':2, 'D:':3, 'D#':4, 'E:':5, 'F:':6, 'F#':7, 'G:':8, 'G#':9, 'A:':10, 'A#':11, 'B:':12}
BAR_EV = 192    
timescale_bounds = [3, 8, 15]


def find_midi_files(root_path):
    midi_files = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith('.midi') or filename.endswith('.mid'):
                full_path = os.path.join(dirpath, filename)
                midi_files.append(full_path)
    return midi_files



def write_report(result_dict, out_csv_file):
    df = pd.DataFrame().from_dict(result_dict)
    df = df.append(df.drop(columns=["piece_name"]).agg(['mean']))
    df = df.round(4)
    df.loc['mean', 'piece_name'] = 'DATASET_MEAN'
    df.to_csv(out_csv_file, index=False, encoding='utf-8')

def midi2event(midi_path):
    note_items, tempo_items = read_items(midi_path)
    if len(note_items) == 0:
        print("empty!")
    note_items = quantize_items(note_items)
    chord_items = extract_chords(note_items)
    items = chord_items + tempo_items + note_items
    max_time = note_items[-1].end
    groups = group_items(items, max_time)
    events = item2event(groups)
    return events




def compute_all_metrics(midi_dir, out_csv, plot_dir=None):
    midi_files = find_midi_files(midi_dir)

    if plot_dir:
        pass
    # result_dict = {
    #   'piece_name': [],
    #   'H1': [],
    #   'H4': [],
    #   'GS': [],
    #   'CPI': []
    # #   'SI_short': [],
    # #   'SI_mid': [],
    # #   'SI_long': []    
    # }

    # for filename, file_plot in zip(sorted(os.listdir(midi_dir)), sorted(os.listdir(plot_dir))):
    #   midi_file = os.path.join(
    #     midi_dir,
    #     filename
    #   )
    #   plot_file = os.path.join(
    #       plot_dir, 
    #       file_plot
    #   )
    #   print(midi_file)
    #   note_items, tempo_items = read_items(midi_file)
    #   note_items = quantize_items(note_items)
    #   chord_items = extract_chords(note_items)
    #   items = chord_items + tempo_items + note_items
    #   max_time = note_items[-1].end
    #   groups = group_items(items, max_time)
    #   events = item2event(groups)
    #   df = []
    #   chord_df = []

    #   for ev in range(len(events)):
    #     ev_name = events[ev:ev+1][0].name
    #     if ev_name == 'Bar':
    #       df.append(BAR_EV)
    #     elif ev_name == 'Note On':
    #       df.append(events[ev:ev+1][0].value)
    #     elif ev_name == 'Position':
    #       df.append(int(Fraction(events[ev:ev+1][0].value)*16)+192)
    #     elif ev_name == 'Chord':
    #       if events[ev:ev+1][0].value[0:2] in CHORD_DICT:
    #         chord_df.append(CHORD_DICT[events[ev:ev+1][0].value[0:2]]+333)


    #   result_dict['piece_name'].append(dir+filename)

    #   h1 = compute_piece_pitch_entropy(df, 1)
    #   h4 = compute_piece_pitch_entropy(df, 4)

    #   result_dict['H1'].append(h1)
    #   result_dict['H4'].append(h4)

    #   groove = compute_piece_groove_similarity(df)
    #   result_dict['GS'].append(groove)

    #   uniq_chord = compute_piece_chord_progression_irregularity(chord_df)
    #   result_dict['CPI'].append(uniq_chord)

    #   si_short = compute_structure_indicator(plot_file, timescale_bounds[0], timescale_bounds[1])
    #   result_dict['SI_short'].append(si_short)
    #   si_mid = compute_structure_indicator(plot_file, timescale_bounds[1], timescale_bounds[2])
    #   result_dict['SI_mid'].append(si_mid)
    #   si_long = compute_structure_indicator(plot_file, timescale_bounds[2])
    #   result_dict['SI_long'].append(si_long)
    else:
        result_dict = {
            'piece_name': [],
            'H1': [],
            'H4': [],
            'GS': [],
            'CPI': []
        }

        for midi_file in sorted(midi_files):
            # midi_file = os.path.join(
            #     midi_dir,
            #     filename
            # )
      
            print(midi_file)
            try:
                note_items, tempo_items = read_items(midi_file)
                note_items = quantize_items(note_items)
                chord_items = extract_chords(note_items)
                items = chord_items + tempo_items + note_items
                max_time = note_items[-1].end
                groups = group_items(items, max_time)
                events = item2event(groups)
                df = []
                chord_df = []

                for ev in range(len(events)):
                    ev_name = events[ev:ev+1][0].name
                    if ev_name == 'Bar':
                        df.append(BAR_EV)
                    elif ev_name == 'Note On':
                        df.append(events[ev:ev+1][0].value)
                    elif ev_name == 'Position':
                        df.append(int(Fraction(events[ev:ev+1][0].value)*16)+192)
                    elif ev_name == 'Chord':
                        if events[ev:ev+1][0].value[0:2] in CHORD_DICT:
                            chord_df.append(CHORD_DICT[events[ev:ev+1][0].value[0:2]]+333)

                h1 = compute_piece_pitch_entropy(df, 1)
                h4 = compute_piece_pitch_entropy(df, 4)
                groove = compute_piece_groove_similarity(df)
                uniq_chord = compute_piece_chord_progression_irregularity(chord_df)
                
                temp_list = []
                for score in [midi_file, h1, h4, groove, uniq_chord]:
                    if score:
                        temp_list.append(score)
                    else:
                        temp_list.append(None)
                result_dict['piece_name'].append(midi_file)
                result_dict['H1'].append(h1)
                result_dict['H4'].append(h4)
                result_dict['GS'].append(groove)
                result_dict['CPI'].append(uniq_chord)
                
            except:
                pass



    if len(result_dict):
        for key, value in result_dict.items():
            print(len(value))
        write_report(result_dict, out_csv)
    else:
        print ('No pieces are found !!')





compute_all_metrics(midi_dir, out_csv)

