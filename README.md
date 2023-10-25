# llama2music


GPU list

id   3 4 5
real 0 1 2

# enviroment
source activate /home/feiyuehchen/anaconda3/envs/llama2
cd /home/feiyuehchen/personality/music_dataset/scripts


# training

export CUDA_VISIBLE_DEVICES=2

screen -L -Logfile screen.log \
python llama_finetuning.py --use_peft \
  --peft_method lora \
  --quantization \
  --model_name ../llama-2-7b-hf \
  --output_dir music/Irish_lora/v1 \
  --dataset alpaca_dataset \
  --batch_size_training 40 \
  --num_epochs 1

# inference
cd /home/feiyuehchen/personality/llama/llama-recipes
python generate.py \
  --base_model ../llama-2-7b-hf \
  --lora_weights music/Irish_lora/v1 


# MU-llama
https://colab.research.google.com/drive/1trpI3zbyLa45ATgUCRGr4VKuqmR0mvpr#scrollTo=lVZfGbS0jVek

cd /home/feiyuehchen/personality/music_dataset/MU-LLaMA/MU-LLaMA
python inference.py \
  --audio_path /home/feiyuehchen/personality/music_dataset/irishman/wav/validation/2122.wav \
  --model ./ckpts/checkpoint.pth \
  --llama_dir /home/feiyuehchen/personality/music_dataset/MU-LLaMA/MU-LLaMA/ckpts/LLaMA

