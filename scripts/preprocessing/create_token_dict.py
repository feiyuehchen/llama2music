import miditok
import os

import argparse
from miditok import TokenizerConfig, REMI, MIDILike, TSD, Structured, CPWord, Octuple, MuMIDI, MMM, MIDITokenizer
import yaml

TOKEN_TYPE = {
    'REMI': REMI, 
    'MIDILike': MIDILike, 
    'TSD': TSD, 
    'Structured': Structured, 
    'CPWord': CPWord, 
    'Octuple': Octuple, 
    'MuMIDI': MuMIDI, 
    'MMM': MMM,
}
NOTE_KEY = {
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'D#',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#',
    9: 'A',
    10: 'A#',
    11: 'B'
}

# TOKEN_CONFIG = {
#     'REMI': TokenizerConfig(use_chords=True, 
#                             use_rests=True,
#                             use_tempos=True,
#                             use_sustain_pedals=True,
#                             use_pitch_bends=True
#                             ), 
#     'MIDILike': MIDILike, 
#     'TSD': TSD, 
#     'Structured': Structured, 
#     'CPWord': CPWord, 
#     'Octuple': Octuple, 
#     'MuMIDI': MuMIDI, 
#     'MMM': MMM,
# }

# Our parameters
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "nb_velocities": 32,
    "special_tokens": [],
    "use_chords": True,
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": False,
    "use_programs": True,
    "use_pitch_bends": True,
    "nb_tempos": 32,  # nb of tempo bins
    "tempo_range": (40, 250),  # (min, max)
}

MAX_BAR_EMBEDDING = {
    "max_bar_embedding": 200

}


def clean_piece(piece):
    value = piece.replace('.', '').lower()
    if 'chord' in value:
        chord, tone = value.split('_')
        if tone[0].isnumeric():
            tone = tone[1:] + tone[0]
        value = tone+chord
    if 'none' in value:
        value = value.replace('none', '')
    if 'timeshift' in value:
        value = value.replace('timeshift', 'shift')
    if 'noteon' in value or 'noteoff' in value or 'pitch' in value:
        pitch, note_number = value.split('_')
        # transform MIDI note number into Note names
        # https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
        # 21 A0
        # 108 C8
        # 109 C#8
        
        octave = (int(note_number)-24)//12+1
        key = NOTE_KEY[(int(note_number)-24)%12]
        value = f"{pitch}{key}{octave}"
    
    value = value.replace('_', '')
    # value = " " + value
    return value

def generate_token_dict(args, MIDI_token):
    print(f'Strat generate {MIDI_token} dictionary...')

    if MIDI_token == 'Octuple':
        config = TokenizerConfig(**TOKENIZER_PARAMS, **MAX_BAR_EMBEDDING)  
    else:
        config = TokenizerConfig(**TOKENIZER_PARAMS)
    tokenizer = TOKEN_TYPE[MIDI_token](config)
    
    output_dict = {}
    if MIDI_token in ['REMI', 'MIDILike', 'TSD', 'Structured', 'MMM']:
        for piece in list(tokenizer.vocab.keys()):
            value = clean_piece(piece)
            output_dict[piece] = value
        
    
    elif MIDI_token in ['CPWord', 'Octuple', 'MuMIDI']:
        for token_dict in tokenizer.vocab:
            for piece in list(token_dict.keys()):
                value = clean_piece(piece)
                output_dict[piece] = value
                

    with open(f"{args.save_dir}/{MIDI_token}.yaml", 'w') as outfile:
        yaml.dump(output_dict, outfile, default_flow_style=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir", default='../../dictionary' ,
        help="path for the root directory for the dictionary.",
    )
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    token_type_list = ['REMI', 'MIDILike', 'TSD', 'Structured', 'MMM', 'CPWord', 'Octuple', 'MuMIDI'] 
    
    for midi_token in token_type_list:
        generate_token_dict(args, midi_token)
    
    