import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
from transformers import LlamaTokenizer

import sentencepiece as spm
import argparse
import yaml
parser = argparse.ArgumentParser()
parser.add_argument('--llama_tokenizer_dir', default=None, type=str, required=True)
parser.add_argument('--dict_path', default='../../dictionary/REMI.yaml', type=str)
parser.add_argument('--output_dir', default='../../tokenizer/planB', type=str)
args = parser.parse_args()

llama_tokenizer_dir = args.llama_tokenizer_dir
with open(args.dict_path, 'r') as f:
    token_dict = yaml.safe_load(f)

# load
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)



token_list = []
for _ , value in token_dict.items():
    piece = f"<{value}>"
    token_list.append(piece)

llama2music_tokens = llama_tokenizer.add_tokens(token_list)

## Save
llama_tokenizer.save_pretrained(args.output_dir)
print(f"llama2music tokenizer has been saved to {args.output_dir}")

# Test
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
llama2music_tokenizer = LlamaTokenizer.from_pretrained(args.output_dir)
print(f"llama2 tokenizer: {len(llama_tokenizer)}")
print(f"llama2music(plabB) tokenizer: {len(llama2music_tokenizer)}")

# print(llama2music_tokenizer)
# print(llama2music_tokenizer.all_special_tokens)
# print(llama2music_tokenizer.all_special_ids)
# print(llama2music_tokenizer.special_tokens_map)
text="<bar><position0><tempo12129><program0><pitchG1><velocity59><duration308><position24><program0><pitchD2><velocity59><duration048><rest048><position0><program0><pitchG2><velocity59><duration108><position8><program0><pitchG2><velocity59><duration208><position24><program0><pitchE2><velocity59><duration348><rest104><position28><program0><pitchB1><velocity59><duration048><bar><position0><program0><pitchB2><velocity59><duration108><rest1202><rest402><position8><program0><pitchD2><velocity59><duration048><rest402><position12><program0><pitchG2><velocity59><duration108><position20><program0><pitchG2><velocity59><duration208><bar><position4><program0><pitchD2><velocity59><duration028><position6><program0><pitchB1><velocity59><duration028><position8><program0><pitchG3><velocity59><duration824><rest1202><rest1202><rest802><position12><program0><pitchB3><velocity59><duration824><bar><bar><position16><program0><pitchG4><velocity59><duration604><bar><bar><position0><program0><pitchF5><velocity59><duration308><rest402><position24><program0><pitchF5><velocity59><duration308><bar><position16><program0><pitchB4><velocity59><duration604><bar><bar><position0><program0><pitchB3><velocity59><duration824><bar><bar><position4><program0><pitchG4><velocity59><duration604><bar><position20><program0><pitchE5><velocity59><duration404><bar><position20><program0><pitchF#4><velocity59><duration604><bar><bar><position4><program0><pitchC4><velocity59><duration604><rest202><position4><program0><pitchG3><velocity59><duration604><rest202><position4><program0><pitchB3><velocity59><duration804><bar><bar><position4><program0><pitchF#3><velocity59><duration404><bar><position4><program0><pitchG3><velocity59><duration804><bar><bar><position4><program0><pitchB3><velocity59><duration804><bar><bar><position4><program0><pitchG3><velocity59><duration028><position6><program0><pitchC4><velocity59><duration804><bar><bar><position6><program0><pitchD4><velocity59><duration804><bar><bar><position6><program0><pitchE2><velocity59><duration108><position14><program0><pitchB1><velocity59><duration048><rest402><position18><program0><pitchG3><velocity59><duration804><bar><bar><position18><program0><pitchD2><velocity59><duration048><position22><program0><pitchG1><velocity59><duration048><position26><program0><pitchE2><velocity59><duration348><bar><position22><program0><pitchE2><velocity59><duration108><position30><program0><pitchC2><velocity59><duration348><bar><position26><program0><pitchG2><velocity59><duration404><bar><position26><program0><pitchC2><velocity59><duration108><bar><position2><program0><pitchF#2><velocity59><duration308><rest402><position26><program0><pitchE4><velocity59><duration108>"

llama_text = text.strip('<').strip('>').replace('><', ' ')
print(f"blank test text: {llama_text}")
print(f"Tokenized by LLaMA tokenizer:{llama_tokenizer.tokenize(llama_text)}")
print("Test text: \n", text)
print(f"Tokenized by llama2music(planB) tokenizer:{llama2music_tokenizer.tokenize(text)}")

print("text REMI token length: 262")

print(f"Tokenized by LLaMA tokenizer:{len(llama_tokenizer.tokenize(llama_text))}")
print(f"Tokenized by llama2music(planB) tokenizer:{len(llama2music_tokenizer.tokenize(text))}")

