import yaml
import json
from miditok import REMI, TokenizerConfig, TokSequence
from miditoolkit import MidiFile
import os
import argparse
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dict_path", default= "../../dictionary/REMI.yaml"
)
parser.add_argument(
    "--data_path", default= "../training/OUTPUT/all_peft_C_REMI_4096_64_128_epoch_5.json"
)
parser.add_argument(
    "--save_dir", default= "../../generation"
)

args = parser.parse_args()
save_dir = os.path.join(args.save_dir, os.path.split(args.data_path)[1].strip('.json'))
os.makedirs(save_dir, exist_ok=True)

def split_tokens(string):
    string = string.replace('\n', '')
    string = string.replace('> <', '><') # planB
    count1 = string.count('><') # planC
    count2 = string.count(' ') # planA

    if count1 > count2:
        return string.strip("<").strip(">").split("><")
    else:
        return string.split(' ')



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


# get reverse dictionary 
with open(args.dict_path, 'r') as f:
    token_dict = yaml.safe_load(f)
token_dict = {v: k for k, v in token_dict.items()}

midi_tokens = []
error_tokens = []

with open(args.data_path, 'r') as f:
    data = f.read()


data = f"[{data.replace('][', '],[')}]"


tmp_data_path = args.data_path.replace('.json', 'tmp.json')
with open(tmp_data_path, 'w') as f:
    f.write(data)

config = TokenizerConfig(**TOKENIZER_PARAMS)
tokenizer = REMI(config)

with open(tmp_data_path) as f:
    dataset = json.load(f)

# print(dataset[-1][108]['output'])
# dataset = dataset[-1]
print()

all_data = []
for id, data in enumerate(dataset[0]):
    try:
        try:
            tokens_str = data['output'].split("Response:")[1] 
        except:
            tokens_str = data['output']
        tokens = split_tokens(tokens_str)

        midi_tokens = []
        error_tokens = []
        for token in tokens:
            try:
                midi_token = token_dict[token]
                midi_tokens.append(midi_token)
            except:
                error_tokens.append(token)


        converted_back_midi = tokenizer(TokSequence(midi_tokens))

        converted_back_midi.dump(os.path.join(save_dir, f"{id}.mid"))

        all_data.append({
            "midi_tokens": len(midi_tokens),
            "error_tokens": len(error_tokens)
        })
    except:
        all_data.append({
            "midi_tokens": None,
            "error_tokens": tokens_str
        })


with open(os.path.join(save_dir, 'temp.json'), 'w') as f:
    json.dump(all_data, f, indent=4)











