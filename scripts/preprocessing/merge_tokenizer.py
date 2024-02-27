import miditoolkit

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
from transformers import LlamaTokenizer

import sentencepiece as spm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--llama_tokenizer_dir', type=str, required=True)
parser.add_argument('--MIDI_BPE_model_path', default='../../tokenizer/piano_data/REMI_20000.model', type=str)
parser.add_argument('--output_dir', default='../../tokenizer/ailab17k_data/planC', type=str)
args = parser.parse_args()

# load

llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_tokenizer_dir)
MIDI_BPE_model = spm.SentencePieceProcessor(model_file = args.MIDI_BPE_model_path)
vocabs = [MIDI_BPE_model.id_to_piece(id) for id in range(MIDI_BPE_model.get_piece_size())]

print(f"Original tokenizer size: {len(llama_tokenizer)}")
print(f"MIDI BPE model sieze: {len(vocabs)}")

llama2music_tokens = llama_tokenizer.add_tokens(vocabs)

## Save
llama_tokenizer.save_pretrained(args.output_dir)
print(f"llama2music tokenizer has been saved to {args.output_dir}")

# Test
llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_tokenizer_dir)
llama2music_tokenizer = LlamaTokenizer.from_pretrained(args.output_dir)
print(f"llama2 tokenizer: {len(llama_tokenizer)}")
print(f"llama2music(plabB) tokenizer: {len(llama2music_tokenizer)}")

# print(llama2music_tokenizer)
# print(llama2music_tokenizer.all_special_tokens)
# print(llama2music_tokenizer.all_special_ids)
# print(llama2music_tokenizer.special_tokens_map)
text="<bar><position0><tempo12129><program0><pitchG1><velocity59><duration308><position24><program0><pitchD2><velocity59><duration048><rest048><position0><program0><pitchG2><velocity59><duration108><position8><program0><pitchG2><velocity59><duration208><position24><program0><pitchE2><velocity59><duration348><rest104><position28><program0><pitchB1><velocity59><duration048><bar><position0><program0><pitchB2><velocity59><duration108><rest1202><rest402><position8><program0><pitchD2><velocity59><duration048><rest402><position12><program0><pitchG2><velocity59><duration108><position20><program0><pitchG2><velocity59><duration208><bar><position4><program0><pitchD2><velocity59><duration028><position6><program0><pitchB1><velocity59><duration028><position8><program0><pitchG3><velocity59><duration824><rest1202><rest1202><rest802><position12><program0><pitchB3><velocity59><duration824><bar><bar><position16><program0><pitchG4><velocity59><duration604><bar><bar><position0><program0><pitchF5><velocity59><duration308><rest402><position24><program0><pitchF5><velocity59><duration308><bar><position16><program0><pitchB4><velocity59><duration604><bar><bar><position0><program0><pitchB3><velocity59><duration824><bar><bar><position4><program0><pitchG4><velocity59><duration604><bar><position20><program0><pitchE5><velocity59><duration404><bar><position20><program0><pitchF#4><velocity59><duration604><bar><bar><position4><program0><pitchC4><velocity59><duration604><rest202><position4><program0><pitchG3><velocity59><duration604><rest202><position4><program0><pitchB3><velocity59><duration804><bar><bar><position4><program0><pitchF#3><velocity59><duration404><bar><position4><program0><pitchG3><velocity59><duration804><bar><bar><position4><program0><pitchB3><velocity59><duration804><bar><bar><position4><program0><pitchG3><velocity59><duration028><position6><program0><pitchC4><velocity59><duration804><bar><bar><position6><program0><pitchD4><velocity59><duration804><bar><bar><position6><program0><pitchE2><velocity59><duration108><position14><program0><pitchB1><velocity59><duration048><rest402><position18><program0><pitchG3><velocity59><duration804><bar><bar><position18><program0><pitchD2><velocity59><duration048><position22><program0><pitchG1><velocity59><duration048><position26><program0><pitchE2><velocity59><duration348><bar><position22><program0><pitchE2><velocity59><duration108><position30><program0><pitchC2><velocity59><duration348><bar><position26><program0><pitchG2><velocity59><duration404><bar><position26><program0><pitchC2><velocity59><duration108><bar><position2><program0><pitchF#2><velocity59><duration308><rest402><position26><program0><pitchE4><velocity59><duration108>"
print("Test text: \n", text)
llama_text = text.strip('<').strip('>').replace('><', ' ')
print(f"blank test text: {llama_text}")
print(f"Tokenized by LLaMA (planA) tokenizer:{llama_tokenizer.tokenize(llama_text)}")
print(f"Tokenized by llama2music (planC) tokenizer:{llama2music_tokenizer.tokenize(text)}")

print("text REMI token length: 262")

print(f"Tokenized by LLaMA (planA) tokenizer:{len(llama_tokenizer.tokenize(llama_text))}")
print(f"Tokenized by llama2music (planB) tokenizer:{263}")
print(f"Tokenized by llama2music(planC) tokenizer:{len(llama2music_tokenizer.tokenize(text))}")

