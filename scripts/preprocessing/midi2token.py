import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
from transformers import pipeline

from miditok import TokenizerConfig, REMI, MIDILike, TSD, Structured, CPWord, Octuple, MuMIDI, MIDITokenizer
from miditoolkit import MidiFile
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import yaml
import random

# musecoco
from musecoco.file_list import generate_file_list
from musecoco.config import attribute_list
import musecoco.midi_data_extractor as mde
from musecoco.midi_data_extractor.verbalizer import Verbalizer

# parallel
from multiprocess import Process, Pool


TOKEN_TYPE = {
    'REMI': REMI, 
    'MIDILike': MIDILike, 
    'TSD': TSD, 
    'Structured': Structured, 
    'CPWord': CPWord, 
    'Octuple': Octuple, 
    'MuMIDI': MuMIDI
}

TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "nb_velocities": 32,
    "special_tokens": [],
    "use_chords": True,
    "use_rests": True,
    "use_tempos": True,
    "use_programs": True,
    "use_pitch_bends": True,
    "nb_tempos": 32,  # nb of tempo bins
    "tempo_range": (40, 250),  # (min, max)
}

MAX_BAR_EMBEDDING = {
    "max_bar_embedding": 200
}


# def process_control_text_and_instruction(midi_paths, error_dict, dataset, lpmc_json_path = None):
#     print("process_control_text_and_instruction")
#     toxigen_roberta = pipeline("text-classification", model="tomh/toxigen_roberta")
#     if dataset == 'lpmc':
#         try:
#             with open(lpmc_json_path, 'r') as f:  
#                 lpmc_json = json.load(f)
#         except:
#             print("Error: You set dataset as lpmc but lpmc_json is None. You have to specify --lpmc_json")
#             return
#     elif dataset == 'musecoco':
#         extractor = mde.DataExtractor(None, encoding_method='REMIGEN2', attribute_list=attribute_list)
#         verbalizer = Verbalizer()
#         random.seed(20)
#      # get control text and insturction
#     control_text_and_instruction_dict = dict()
#     for midi_path in tqdm(midi_paths):
#         try:
#             midi = MidiFile(midi_path)
#         except:
#             error_dict[str(midi_path)] = 'cannot read midi file'
#         if not midi.instruments:
#             error_dict[str(midi_path)] = 'no midi instruments'
#             continue   
        
#         data = dict()
#         if dataset == 'lyrics2midi':
#             lyrics = ' '.join([marker.text for marker in midi.markers])
#             if toxigen_roberta(lyrics[:512])[0]['label'] == "LABEL_1":
#                 error_dict[str(midi_path)] = 'detect toxic in lyrics'
#                 continue

#             data["control_code"] = f"Lyrics: {lyrics}"
#             data["instruction"] = "Please provide me with music based on the following lyrics."
#         elif dataset == 'lpmc':
#             # get the longest sentence in the dictionary
#             if '.midi' in str(midi_path):
#                 wav_name = os.path.split(midi_path)[1].replace('.midi', '.wav')
#             elif '.mid' in str(midi_path):
#                 wav_name = os.path.split(midi_path)[1].replace('.mid', '.wav')
#             if wav_name in lpmc_json:
#                 control_text = max(list(lpmc_json[wav_name].values()), key = len)
#                 data["control_code"] = f"Text: {control_text}"
#                 data["instruction"] = "Please provide me with music based on the following music description."
#             else:
#                 error_dict[str(midi_path)] = f'wav file name: {wav_name} is not in music2text json file!'
#                 continue
#         elif dataset == 'musecoco':
#             try:
#                 _, _, _, info_dict, _ = extractor.extract(
#                     os.path.split(midi_path)[0], os.path.split(midi_path)[1],
#                     cut_method='random_2',
#                     normalize_pitch_value=True,
#                     # === load values for the subjective attributes here ===
#                     artist=None,  # 'mozart',
#                     genre=None,  # ('Pop_Rock', 'RnB'), 
#                     emotion=None,  # 'Q1',
#                     # =============
#                 )

#                 pieces = info_dict['pieces']
#                 midi_text_list = []
#                 # musecoco support multiple instruments
#                 # however, in llama2music, the music we use here are pop piano

#                 for piece in pieces:
#                     value_dict = piece['values']
#                     piece_text_list, _ = verbalizer.get_text(value_dict)
#                     # print(piece_text_list)
#                     midi_text_list.append(random.choices(piece_text_list))
                    
#                 # if you want to use multiple instruments
#                 # data["control_code"] = '\n'.join(midi_text_list)
#                 # if you only want single instument
#                 data["control_code"] = f"Text: {midi_text_list[0][0]}"
#                 data["instruction"] = "Please provide me with music based on the following music attributes."
#             except: 
#                 error_dict[str(midi_path)] = "cannot get music attributes"
#                 # raise
#         else:
#             data["control_code"] = None
#             data["instruction"] = "Please provide me with music."

#         midiname = os.path.split(midi_path)[1]
#         control_text_and_instruction_dict[midiname] = data
        

#     return control_text_and_instruction_dict, error_dict

def process_midi_path(midi_path, dataset, toxic_model, extractor, verbalizer, lpmc_json=None):
    error_dict = {}
    data = {}
    try:
        midi = MidiFile(midi_path)
    except:
        error_dict[str(midi_path)] = 'cannot read midi file'
        return midi_path, data, error_dict
    if not midi.instruments:
        error_dict[str(midi_path)] = 'no midi instruments'
        return midi_path, data, error_dict
    

    if dataset == 'lyrics2midi':
        lyrics = ' '.join([marker.text for marker in midi.markers])
        if toxic_model(lyrics[:512])[0]['label'] == "LABEL_1":
            error_dict[str(midi_path)] = 'detect toxic in lyrics'
            return midi_path, data, error_dict

        data["control_code"] = f"Lyrics: {lyrics}"
        data["instruction"] = "Please provide me with music based on the following lyrics."
    elif dataset == 'lpmc':
        if '.midi' in str(midi_path):
            wav_name = os.path.split(midi_path)[1].replace('.midi', '.wav')
        elif '.mid' in str(midi_path):
            wav_name = os.path.split(midi_path)[1].replace('.mid', '.wav')
        if wav_name in lpmc_json:
            control_text = max(list(lpmc_json[wav_name].values()), key=len)
            data["control_code"] = f"Text: {control_text}"
            data["instruction"] = "Please provide me with music based on the following music description."
        else:
            error_dict[str(midi_path)] = f'wav file name: {wav_name} is not in music2text json file!'
            return midi_path, data, error_dict
    elif dataset == 'musecoco':
        try:
            _, _, _, info_dict, _ = extractor.extract(
                os.path.split(midi_path)[0], os.path.split(midi_path)[1],
                cut_method='random_2',
                normalize_pitch_value=True,
                # === load values for the subjective attributes here ===
                artist=None,  # 'mozart',
                genre=None,  # ('Pop_Rock', 'RnB'), 
                emotion=None,  # 'Q1',
                # =============
            )

            pieces = info_dict['pieces']
            midi_text_list = []
            # musecoco support multiple instruments
            # however, in llama2music, the music we use here are pop piano

            for piece in pieces:
                value_dict = piece['values']
                piece_text_list, _ = verbalizer.get_text(value_dict)
                # print(piece_text_list)
                midi_text_list.append(random.choices(piece_text_list))
                
            # if you want to use multiple instruments
            # data["control_code"] = '\n'.join(midi_text_list)
            # if you only want single instument
            data["control_code"] = f"Text: {midi_text_list[0][0]}"
            data["instruction"] = "Please provide me with music based on the following music attributes."
        except: 
            error_dict[str(midi_path)] = "cannot get music attributes"
    else:
        data["control_code"] = None
        data["instruction"] = "Please provide me with music."

    return midi_path, data, error_dict

def process_control_text_and_instruction(midi_paths, error_dict, dataset, lpmc_json_path=None):
    if dataset == 'lpmc':
        print("lpmc")
        try:
            with open(lpmc_json_path, 'r') as f:
                lpmc_json = json.load(f)
        except:
            print("Error: You set dataset as lpmc but lpmc_json is None. You have to specify --lpmc_json")
            return

    pool = Pool(32)  
    toxigen_roberta = pipeline("text-classification", model="tomh/toxigen_roberta")
    extractor = mde.DataExtractor(None, encoding_method='REMIGEN2', attribute_list=attribute_list)
    verbalizer = Verbalizer()
    random.seed(20)
    results = pool.starmap(process_midi_path, [(midi_path, 
                                                dataset, 
                                                toxigen_roberta if dataset == 'lyrics2midi' else None,
                                                extractor if dataset == 'musecoco' else None,
                                                verbalizer if dataset == 'musecoco' else None,
                                                lpmc_json if dataset == 'lpmc' else None) for midi_path in midi_paths])
    pool.close()
    pool.join()

    control_text_and_instruction_dict = {}
    for result in results:
        midi_path, data, err = result
        if data:
            midiname = os.path.split(midi_path)[1]
            control_text_and_instruction_dict[midiname] = data
        if err:
            error_dict.update(err)

    return control_text_and_instruction_dict, error_dict


# def midi2json(args, midi_paths, dataset_type, control_text_and_instruction_dict, error_dict, midi_token=None, blank=None):
#     print('midi2json')
#     if midi_token is None:
#         midi_token = args.MIDI_token
#     dict_path = f"{args.token_dict_dir}/{midi_token}.yaml"
#     if os.path.exists(dict_path):
#         with open(dict_path, 'r') as f:
#             token_dict = yaml.safe_load(f)
#     else:
#         token_dict = dict()

#     if blank is None:
#         blank = args.add_blank

#     print(f"Type of the Token: {midi_token}")
#     print(f"Type of the blank: {blank}")
#     output_json = []
            
#     # Creating a multitrack tokenizer configuration
#     if midi_token == 'Octuple':
#         config = TokenizerConfig(**TOKENIZER_PARAMS, **MAX_BAR_EMBEDDING)  
#     else:
#         config = TokenizerConfig(**TOKENIZER_PARAMS)
#     tokenizer = TOKEN_TYPE[midi_token](config)
    
    
#     # process midi
#     for midi_path in tqdm(midi_paths):
#         try:
#             midi = MidiFile(midi_path)
#         except:
#             error_dict[str(midi_path)] = 'cannot read midi file'
#             continue

#         if not midi.instruments:
#             error_dict[str(midi_path)] = "no midi instruments"
#             continue   
#         # try:
#         if midi_token in ['REMI', 'MIDILike', 'TSD', 'Structured']:
#             try:
#                 tokens = tokenizer(midi)
#             except:
#                 error_dict[str(midi_path)] = "midi cannot be tokenized"
#                 continue
#             midi_token_list = []
#             for token in tokens.tokens:
#                 midi_token_list.append(token_dict[token])
#             if blank:
#                 midi_text = ' '.join(midi_token_list)
#             else:
#                 midi_token_list = [f"<{token}>" for token in midi_token_list]
#                 midi_text = ''.join(midi_token_list)
                
#             midiname = os.path.split(midi_path)[1]
#             data = {
#                 "midiname": midiname,
#                 "dataset_type": dataset_type,
#                 "token_type": midi_token,
#                 "token_length": len(tokens.tokens),
#                 "tokens": midi_text
#             }
#         elif midi_token in ['CPWord', 'Octuple', 'MuMIDI']:
#             midi_text = []
#         # here, we not only turn midi into tokens, but also add new tokens to the dictionary

#             if blank:
#                 for token_list in tokens.tokens:
#                     midi_text.append(' '.join([token_dict[token] for token in token_list]))
#             else:
#                 for token_list in tokens.tokens:
#                     midi_text.append(''.join([token_dict[token] for token in token_list]))
#             midi_text = '\n'.join(midi_text)
#             midiname = os.path.split(midi_path)[1]
#             data = {
#                 "midiname": midiname,
#                 "dataset_type": dataset_type,
#                 "token_type": midi_token,
#                 "token_length": len(tokens[0].tokens),
#                 "tokens": midi_text
#             }
#         # except:
#         #     error_dict[str(midi_path)] = "cannot turn midi into tokens"
#         #     continue

#         # add control text and instruction
#         try:
#             data["instruction"] =  control_text_and_instruction_dict[midiname]["instruction"]
#             data["control_code"] = control_text_and_instruction_dict[midiname]["control_code"]
#         except:
#             error_dict[str(midi_path)] = "no control code or instruction"

#         output_json.append(data)
    
    # #save json files
    # print(f"Number of json files: {len(output_json)}")
    # print(f"Number of error files: {len(error_dict.items())}")
    # return output_json, error_dict

def process_midi(midi_path, midi_token, token_dict, blank, dataset_type, control_text_and_instruction_dict, error_dict):
    data = {}
    try:
        midi = MidiFile(midi_path)
    except:
        error_dict[str(midi_path)] = 'cannot read midi file'
        return data

    if not midi.instruments:
        error_dict[str(midi_path)] = "no midi instruments"
        return data

    try:
        tokens = tokenizer(midi)
    except:
        error_dict[str(midi_path)] = "midi cannot be tokenized"
        return data

    midi_token_list = []
    for token in tokens.tokens:
        midi_token_list.append(token_dict[token])
    if blank:
        midi_text = ' '.join(midi_token_list)
    else:
        midi_token_list = [f"<{token}>" for token in midi_token_list]
        midi_text = ''.join(midi_token_list)

    midiname = os.path.split(midi_path)[1]
    data = {
        "midiname": midiname,
        "dataset_type": dataset_type,
        "token_type": midi_token,
        "token_length": len(tokens.tokens),
        "tokens": midi_text
    }

    # Add control text and instruction
    try:
        data["instruction"] = control_text_and_instruction_dict[midiname]["instruction"]
        data["control_code"] = control_text_and_instruction_dict[midiname]["control_code"]
    except:
        error_dict[str(midi_path)] = "no control code or instruction"

    return data

def midi2json(args, midi_paths, dataset_type, control_text_and_instruction_dict, error_dict, midi_token=None, blank=None):
    print('midi2json')
    if midi_token is None:
        midi_token = args.MIDI_token
    dict_path = f"{args.token_dict_dir}/{midi_token}.yaml"
    if os.path.exists(dict_path):
        with open(dict_path, 'r') as f:
            token_dict = yaml.safe_load(f)
    else:
        token_dict = dict()

    if blank is None:
        blank = args.add_blank

    print(f"Type of the Token: {midi_token}")
    print(f"Type of the blank: {blank}")
    output_json = []

    # Creating a multitrack tokenizer configuration
    if midi_token == 'Octuple':
        config = TokenizerConfig(**TOKENIZER_PARAMS, **MAX_BAR_EMBEDDING)
    else:
        config = TokenizerConfig(**TOKENIZER_PARAMS)
    global tokenizer
    tokenizer = TOKEN_TYPE[midi_token](config)

    # Create a multiprocessing pool with 32 processes
    pool = Pool(processes=32)

    # Process midi files in parallel
    results = []
    for midi_path in tqdm(midi_paths):
        result = pool.apply_async(process_midi, args=(midi_path, midi_token, token_dict, blank, dataset_type, control_text_and_instruction_dict, error_dict))
        results.append(result)

    pool.close()
    pool.join()

    for result in tqdm(results):
        data = result.get()
        if data:
            output_json.append(data)

    # Save json files
    print(f"Number of json files: {len(output_json)}")
    print(f"Number of error files: {len(error_dict.items())}")
    return output_json, error_dict



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--midi_dir", default = "../../dataset/ailab17k_data/seg_midi_data",
        help="directory of midi files",
    )
    parser.add_argument(
        "--save_dir", default = "../../dataset/ailab17k_data/seg_json_data",
        help="save directory for the output json file"
    )
    parser.add_argument(
        "--json_name", default='data.json',
        help="save name for the output json file"
    )
    parser.add_argument(
        "--token_dict_dir", default='../../dictionary', 
        help="token dictionary directory"
    )
    parser.add_argument(
        "--MIDI_token", default='REMI', 
        choices=['REMI',  'MIDILike', 'TSD', 'Structured', 'CPWord', 'Octuple', 'MuMIDI'],
        help='type of MIDI tokenization'
    )
    parser.add_argument(
        "--dataset", default=None, choices=[None, 'lyrics2midi', 'lpmc', 'musecoco'],
        help="if the dataset has unique procedure"
    )
    parser.add_argument(
        "--lpmc_json", default=None,
        help="if you set the dataset to lpmc, you have to set a lpmc json file"
    )
    parser.add_argument(
        "--all_tokens", default=False,
        help="run all tokens." 
    )
    parser.add_argument(
        "--add_blank", default=False,
        help="for the original tokenizer, adding a blank between midi tokens(processed by dictionary) can get a shorter tokens." 
    )
    parser.add_argument(
        "--multiprocess", default= 32, 
        help="parallel run def process_pretrain_data().",
    )
    parser.add_argument(
        "--run_all_dataset", default=True,
        help="default setting for running the raw directory"
    )

    args = parser.parse_args()

    with open(f"{args.midi_dir}/dataprocess.yaml", 'r') as f:
        DATA_PROCESS = yaml.safe_load(f)
    error_json_dir = os.path.join(args.save_dir, 'error')
    os.makedirs(error_json_dir, exist_ok=True)
    def run_one_dataset(dataset_dir, dataset_type, save_path, lpmc_json_path):
        midi_paths = list(Path(dataset_dir).glob("**/*.mid")) + list(Path(dataset_dir).glob("**/*.midi"))
        error_dict = {}
        control_text_and_instruction_dict, error_dict = process_control_text_and_instruction(midi_paths, error_dict, DATA_PROCESS[dataset_type], lpmc_json_path)
        output_json, error_dict = midi2json(args, midi_paths, dataset_type, control_text_and_instruction_dict, error_dict, args.MIDI_token, args.add_blank)
        print(f"save at: {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(output_json, f, indent=4)
        if len(error_dict.items()):
            with open(os.path.join(error_json_dir, f"{dataset_type}_{DATA_PROCESS[dataset_type]}.json"), 'w') as f:
                json.dump([error_dict], f, indent=4)


    if args.run_all_dataset:
        
        split_folder = ['test', 'validation', 'train']

        for split_type in split_folder:
            data_dir_list = os.listdir(f"{args.midi_dir}/{split_type}")
            for dataset_folder in data_dir_list:
                dataset_dir = f"{args.midi_dir}/{split_type}/{dataset_folder}"
                save_path = f"{args.save_dir}/{split_type}/{dataset_folder}_{DATA_PROCESS[dataset_folder]}.json"
                lmpc_json_path = f"{args.midi_dir}/music2text.json"
                print("="*60)
                print(f"Type: {split_type}")
                print(f"Dataset: {dataset_folder}")
                print(f"Process: {DATA_PROCESS[dataset_folder]}")
                run_one_dataset(dataset_dir, dataset_folder, save_path, lmpc_json_path)



