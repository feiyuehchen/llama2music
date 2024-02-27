# llama2music
export CUDA_VISIBLE_DEVICES=1
python -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization
# enviroment


source activate /home/feiyuehchen/anaconda3/envs/llama2
cd /home/feiyuehchen/personality/llama2music/scripts/preprocessing
python music2text.py --save_path /home/feiyuehchen/personality/llama2music/dataset/piano_data/midi_data/music2text.json --audio_dir /home/feiyuehchen/personality/llama2music/dataset/piano_data/wav_data --npy_dir /home/feiyuehchen/personality/llama2music/dataset/piano_data/npy --lpmc_dir /home/feiyuehchen/personality/music_dataset/lp-music-caps

source deactivate
conda deactivate
source activate /home/feiyuehchen/anaconda3/envs/llama2



python music2text.py --save_path /home/feiyuehchen/personality/llama2music/dataset/ailab17k_data/seg_midi_data/music2text.json --audio_dir /home/feiyuehchen/personality/llama2music/dataset/ailab17k_data/seg_wav_data --npy_dir /home/feiyuehchen/personality/llama2music/dataset/ailab17k_data/seg_npy --lpmc_dir /home/feiyuehchen/personality/music_dataset/lp-music-caps 

cd /home/feiyuehchen/personality/llama2music/scripts/training
# for root
source /home/feiyuehchen/anaconda3/envs/llama2/bin/activate

# sheetstage
source activate /home/feiyuehchen/anaconda3/envs/llama2/envs/jukebox 
cd /home/feiyuehchen/personality/music_dataset/msd_sheet


camel 0~100000
python ../../llama2music/scripts/msd2sheet.py --slice_start 0 --slice_end 100000

neverdie 100000~200000
python ../../llama2music/scripts/msd2sheet.py --slice_start 100000 --slice_end 200000

wolverine 200000~300000
python ../../llama2music/scripts/msd2sheet.py --slice_start 200000 --slice_end 300000 --multiprocess 10

neveroom 300000~400000
python ../../llama2music/scripts/msd2sheet.py --slice_start 300000 --slice_end 400000 --multiprocess 24

coffee 400000~500000
python ../../llama2music/scripts/msd2sheet.py --slice_start 400000 --slice_end 500000 



# Create small dataset for Irishman
token:
  REMI, MIDILike, TSD, Structured, CPWord, Octuple, MuMIDI, MMM
cd /home/feiyuehchen/personality/llama2music/scripts/preprocessing
python create_small_dataset.py --data_path /home/feiyuehchen/personality/llama2music/dataset/Irishman/MuMIDI/train.json --save_path /home/feiyuehchen/personality/llama2music/dataset/Irishman/MuMIDI/small_train.json --slice 25000
python create_small_dataset.py --data_path /home/feiyuehchen/personality/llama2music/dataset/Irishman/MuMIDI/validation.json --save_path /home/feiyuehchen/personality/llama2music/dataset/Irishman/MuMIDI/small_validation.json --slice 100







# training
source activate /home/feiyuehchen/anaconda3/envs/llama2
token:
  REMI, MIDILike, TSD, Structured, CPWord, Octuple, MuMIDI, MMM
export CUDA_VISIBLE_DEVICES=1

cd /home/feiyuehchen/personality/llama2music/scripts/training
screen -L -Logfile screen_MuMIDI.log \
python llama_finetuning.py --use_peft \
  --peft_method lora \
  --quantization \
  --model_name /home/feiyuehchen/personality/llama/llama-2-7b-hf \
  --tokenizer /home/feiyuehchen/personality/llama2music/tokenizer/MuMIDI/noBPE/merged_tokenizer_hf \
  --output_dir ../path/to/ckpt/Irishman/MuMIDI/v1 \
  --dataset Irishman_dataset_MuMIDI \
  --batch_size_training 2\
  --micro_batch_size 2\
  --num_epochs 1\
  --use_fast_kernels True





python llama_finetuning.py --use_peft \
  --peft_method lora \
  --quantization \
  --model_name /home/feiyuehchen/personality/llama/llama-2-7b-hf \
  --tokenizer /home/feiyuehchen/personality/llama/llama-2-7b-hf \
  --output_dir ../path/to/ckpt/Irishman/llama/REMI_v1 \
  --dataset Irishman_dataset_REMI \
  --batch_size_training 2\
  --micro_batch_size 2\
  --num_epochs 1


# merge checkpoint and tokenzier
source activate /home/feiyuehchen/anaconda3/envs/llama2

cd /home/feiyuehchen/personality/llama2music/scripts    
template:
python merge_llama2_with_chinese_lora_low_mem.py \
    --base_model path_to_original_llama2_hf_dir \
    --lora_model path_to_chinese_llama2_or_alpaca2_lora \
    --output_type huggingface \
    --output_dir path_to_output_dir \
    --verbose

python merge_llama2_with_chinese_lora_low_mem.py \
    --base_model  /home/feiyuehchen/personality/llama/llama-2-7b-hf\
    --lora_mode  /home/feiyuehchen/personality/llama2music/scripts/path/to/ckpt/Irishman/REMI/v1\
    --output_type huggingface \
    --output_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/output_model/Irishman/REMI/noBPE_v2 \
    --verbose


# inference
export CUDA_VISIBLE_DEVICES=1

cd /home/feiyuehchen/personality/llama2music/scripts/inference
python generate_gradio.py \
  --base_model /home/feiyuehchen/personality/llama/llama-2-7b-hf \
  --lora_weights /home/feiyuehchen/personality/llama2music/scripts/path/to/ckpt/Irishman/REMI/v1\
  --tokenizer_path /home/feiyuehchen/personality/music_dataset/llama2music/tokenizer/REMI/noBPE/merged_tokenizer_hf

python generate.py \
    --base_model /home/feiyuehchen/personality/llama2music/scripts/path/to/output_model/Irishman/REMI/noBPE \
    --with_prompt \
    --data_file input.json\
    --gpus 1

# MU-llama
https://colab.research.google.com/drive/1trpI3zbyLa45ATgUCRGr4VKuqmR0mvpr#scrollTo=lVZfGbS0jVek

cd /home/feiyuehchen/personality/music_dataset/MU-LLaMA/MU-LLaMA
python inference.py \
  --audio_path /home/feiyuehchen/personality/music_dataset/irishman/wav/validation/2122.wav \
  --model ./ckpts/checkpoint.pth \
  --llama_dir /home/feiyuehchen/personality/music_dataset/MU-LLaMA/MU-LLaMA/ckpts/LLaMA



../../llama/llama-2-7b-hf


# tokenizer length


# midi2token
source activate /home/feiyuehchen/anaconda3/envs/llama2
cd /home/feiyuehchen/personality/llama2music/scripts/preprocessing

python midi2token.py --midi_dir /home/feiyuehchen/personality/music_dataset/llama2music/old_dataset/music/raw/LSTM_GAN --save_path /home/feiyuehchen/personality/llama2music/scripts/path/to/LSTM_GAN/REMI/temp.json --dataset LSTM_GAN 

python midi2token.py --midi_dir /home/feiyuehchen/personality/music_dataset/llama2music/old_dataset/music/raw/LSTM_GAN --save_path /home/feiyuehchen/personality/llama2music/scripts/path/to/LSTM_GAN/CPWord/temp.json --dataset LSTM_GAN --MIDI_token CPWord

python merge_tokenizer.py --llama_tokenizer_dir ../../../llama/llama-2-7b-hf --midi_dir ../../../music_dataset/irishman/midi/train --save_path ../path/to/Irishman/REMI/train.json --MIDI_token REMI --dataset Irishman

cd /home/feiyuehchen/personality/llama2music/scripts
python midi2token.py --llama_tokenizer_dir ../../llama/llama-2-7b-hf --midi_dir ../../music_dataset/irishman/midi/train --save_path path/to/Irishman/MMM/train.json --MIDI_token MMM --dataset Irishman


python midi2token.py --llama_tokenizer_dir ../../llama/llama-2-7b-hf --midi_dir ../../music_dataset/irishman/midi/validation --save_path path/to/Irishman/CPWord/validation.json --MIDI_token CPWord --dataset Irishman

bpe

cd /home/feiyuehchen/personality/llama2music/scripts/preprocessing

python merge_tokenizer.py --llama_tokenizer_dir ../../../llama/llama-2-7b-hf --midi_dir ../../../music_dataset/irishman/midi/validation --save_path ../path/to/Irishman/REMI/validation.json --MIDI_token REMI --bpe True --dataset Irishman



# token to scripts

cd /home/feiyuehchen/personality/llama2music/scripts/preprocessing
template:
python token2script.py --save_path ../../dataset/LSTM_GAN/REMI/temp.json  --token_path /home/feiyuehchen/personality/llama2music/dataset/LSTM_GAN/CPWord/temp.json


python token2script.py --save_path ../../dataset/Irishman/REMI/train.json --dataset Irishman --token_path ../path/to/Irishman/REMI/train.json

python token2script.py --save_path ../../dataset/Irishman/REMI/validation.json --dataset Irishman --token_path ../path/to/Irishman/REMI/validation.json


python token2script.py --save_path ../../dataset/Irishman/MMM/train.json --dataset Irishman --token_path ../path/to/Irishman/MMM/train.json
python token2script.py --save_path ../../dataset/Irishman/MMM/validation.json --dataset Irishman --token_path ../path/to/Irishman/MMM/validation.json




Give me an example in ABC notation for the melody with the following control codes. S is the number of sections, B is the number of bars, and E is the edit distance similarity.


Give me an example in ABC notation for the melody with the following control codes.
S:number of sections determines the number of sections in the entire melody. It counts on several symbols that can be used to represent section boundaries:[|,||,|],|:,::, and :|. The range is 1 to 8 (e.g., S:1 and S:8).
B:number of bars specifies the desired number of bars within a section. It counts on the bar symbol |. The range is 1 to 32 (e.g., B:1 and B:32).
E:edit distance similarity controls the similarity level between the current section c and a previous section p in the melody. It is based on the Levenshtein distance lev(c, p) , quantifying the difference between sections for creating variations or contrasts. It can be expressed as:eds(c, p) = 1 − (lev(c, p)/max(|c|, |p|)) where |c| and |p| are the string lengths of the two sections. It is discretized into 11 levels, ranging from no match at all to an exact match (e.g., E:0 and E:10).


Give me an example in ABC notation for the melody with the following control codes:
S:number of sections determines the number of sections in the entire melody. It counts on several symbols that can be used to represent section boundaries:[|,||,|],|:,::, and :|. The range is 1 to 8 (e.g., S:1 and S:8).
B:number of bars specifies the desired number of bars within a section. It counts on the bar symbol |. The range is 1 to 32 (e.g., B:1 and B:32).
E:edit distance similarity controls the similarity level between the current section c and a previous section p in the melody. It is discretized into 11 levels, ranging from no match at all to an exact match (e.g., E:0 and E:10).

"Give me an example in ABC notation for the melody with the following control codes:

S: Number of sections determines the number of sections in the entire melody. It relies on several symbols that can be used to represent section boundaries: [|, ||, |], |:, ::, and :|. The range is 1 to 8 (e.g., S:1 and S:8).

B: Number of bars specifies the desired number of bars within a section. It depends on the bar symbol |. The range is 1 to 32 (e.g., B:1 and B:32).

E: Edit distance similarity controls the similarity level between the current section c and a previous section p in the melody. It is based on the Levenshtein distance lev(c, p), quantifying the difference between sections for creating variations or contrasts. It can be expressed as: eds(c, p) = 1 − (lev(c, p)/max(|c|, |p|)), where |c| and |p| are the string lengths of the two sections. It is discretized into 11 levels, ranging from no match at all to an exact match (e.g., E:0 and E:10)."



objective metrics:
1. musecoco: asa
@article{musecoco2023,
  title={MuseCoco: Generating Symbolic Music from Text},
  author={Peiling Lu, Xin Xu, Chenfei Kang, Botao Yu, Chengyi Xing, Xu Tan, Jiang Bian},
  journal={arXiv preprint arXiv:2306.00110},
  year={2023}
}
2.








python midi2token.py --midi_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/LSTM_GAN/midi/validation --save_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/data/noblank_MIDILike/validation --json_name LSTM_GAN_validation.json --MIDI_token MIDILike --dataset lyrics2midi --plus_reverse True
python midi2token.py --midi_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/LSTM_GAN/midi/validation --save_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/data/noblank_REMI/validation --json_name LSTM_GAN_validation.json --MIDI_token REMI --dataset lyrics2midi --plus_reverse True
python midi2token.py --midi_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/LSTM_GAN/midi/validation --save_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/data/noblank_Structured/validation --json_name LSTM_GAN_validation.json --MIDI_token Structured --dataset lyrics2midi  --plus_reverse True
python midi2token.py --midi_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/LSTM_GAN/midi/validation --save_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/data/noblank_TSD/validation --json_name LSTM_GAN_validation.json --MIDI_token TSD --dataset lyrics2midi --plus_reverse True



python midi2token.py --midi_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/LSTM_GAN/midi/validation --save_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/data/blank_MIDILike/validation --json_name LSTM_GAN_validation.json --MIDI_token MIDILike --dataset lyrics2midi --add_blank True --plus_reverse True
python midi2token.py --midi_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/LSTM_GAN/midi/validation --save_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/data/blank_REMI/validation --json_name LSTM_GAN_validation.json --MIDI_token REMI --dataset lyrics2midi --add_blank True --plus_reverse True
python midi2token.py --midi_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/LSTM_GAN/midi/validation --save_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/data/blank_Structured/validation --json_name LSTM_GAN_validation.json --MIDI_token Structured --dataset lyrics2midi --add_blank True --plus_reverse True
python midi2token.py --midi_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/LSTM_GAN/midi/validation --save_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/data/blank_TSD/validation --json_name LSTM_GAN_validation.json --MIDI_token TSD --dataset lyrics2midi --add_blank True --plus_reverse True


python midi2token.py --midi_dir /home/feiyuehchen/personality/music_dataset/EMOPIA_1-2.0/midis --save_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/data/blank_MIDILike --json_name emopia.json --MIDI_token MIDILike --dataset musecoco --add_blank True --split_to_train_valid True
python midi2token.py --midi_dir /home/feiyuehchen/personality/music_dataset/EMOPIA_1-2.0/midis --save_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/data/blank_REMI --json_name emopia.json --MIDI_token REMI --dataset musecoco --add_blank True --split_to_train_valid True
python midi2token.py --midi_dir /home/feiyuehchen/personality/music_dataset/EMOPIA_1-2.0/midis --save_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/data/blank_Structured --json_name emopia.json --MIDI_token Structured --dataset musecoco --add_blank True --split_to_train_valid True
python midi2token.py --midi_dir /home/feiyuehchen/personality/music_dataset/EMOPIA_1-2.0/midis --save_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/data/blank_TSD --json_name emopia.json --MIDI_token TSD --dataset musecoco --add_blank True --split_to_train_valid True
python midi2token.py --midi_dir /home/feiyuehchen/personality/music_dataset/EMOPIA_1-2.0/midis --save_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/data/noblank_MIDILike --json_name emopia.json --MIDI_token MIDILike --dataset musecoco --split_to_train_valid True 
python midi2token.py --midi_dir /home/feiyuehchen/personality/music_dataset/EMOPIA_1-2.0/midis --save_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/data/noblank_REMI --json_name emopia.json --MIDI_token REMI --dataset musecoco --split_to_train_valid True 
python midi2token.py --midi_dir /home/feiyuehchen/personality/music_dataset/EMOPIA_1-2.0/midis --save_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/data/noblank_Structured --json_name emopia.json --MIDI_token Structured --dataset musecoco --split_to_train_valid True 
python midi2token.py --midi_dir /home/feiyuehchen/personality/music_dataset/EMOPIA_1-2.0/midis --save_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/data/noblank_TSD --json_name emopia.json --MIDI_token TSD --dataset musecoco --split_to_train_valid True



python midi2token.py --midi_dir /home/feiyuehchen/personality/music_dataset/EMOPIA_1-2.0/midis --save_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/data/blank_MIDILike --json_name emopia.json --MIDI_token MIDILike --dataset musecoco --add_blank True --split_to_train_valid True
python midi2token.py --midi_dir /home/feiyuehchen/personality/music_dataset/EMOPIA_1-2.0/midis --save_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/data/blank_REMI --json_name emopia.json --MIDI_token REMI --dataset musecoco --add_blank True --split_to_train_valid True
python midi2token.py --midi_dir /home/feiyuehchen/personality/music_dataset/EMOPIA_1-2.0/midis --save_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/data/blank_Structured --json_name emopia.json --MIDI_token Structured --dataset musecoco --add_blank True --split_to_train_valid True
python midi2token.py --midi_dir /home/feiyuehchen/personality/music_dataset/EMOPIA_1-2.0/midis --save_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/data/blank_TSD --json_name emopia.json --MIDI_token TSD --dataset musecoco --add_blank True --split_to_train_valid True
python midi2token.py --midi_dir /home/feiyuehchen/personality/music_dataset/EMOPIA_1-2.0/midis --save_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/data/noblank_MIDILike --json_name emopia.json --MIDI_token MIDILike --dataset musecoco --split_to_train_valid True 
python midi2token.py --midi_dir /home/feiyuehchen/personality/music_dataset/EMOPIA_1-2.0/midis --save_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/data/noblank_REMI --json_name emopia.json --MIDI_token REMI --dataset musecoco --split_to_train_valid True 
python midi2token.py --midi_dir /home/feiyuehchen/personality/music_dataset/EMOPIA_1-2.0/midis --save_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/data/noblank_Structured --json_name emopia.json --MIDI_token Structured --dataset musecoco --split_to_train_valid True 
python midi2token.py --midi_dir /home/feiyuehchen/personality/music_dataset/EMOPIA_1-2.0/midis --save_dir /home/feiyuehchen/personality/llama2music/scripts/path/to/data/noblank_TSD --json_name emopia.json --MIDI_token TSD --dataset musecoco --split_to_train_valid True