import os
# from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
from datasets import load_dataset, load_from_disk
import argparse
from tqdm import tqdm
from tokenizers import SentencePieceBPETokenizer
import yaml



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="../../dataset/ailab17k_data/processed_data", 
        help="path for the json data, you could run token2script.py to get it"
    )
    parser.add_argument(
        "--temp_dir", default="../../dataset/ailab17k_data/tmp/txt", 
        help="path for saving txt file"
    )
    parser.add_argument(
        "--model_prefix", default="../../tokenizer/ailab17k_data/REMI_20000_128"
    )
    parser.add_argument(
        '--dict_path', default='../../dictionary/REMI.yaml', type=str
    )
    parser.add_argument(
        '--max_sentence_length', default=30000, type=int
    )
    
    
    args = parser.parse_args()
    print(os.path.dirname(args.model_prefix))
    os.makedirs(os.path.dirname(args.model_prefix), exist_ok=True)
    with open(args.dict_path, 'r') as f:
        token_dict = yaml.safe_load(f)
    token_list = []
    for _ , value in token_dict.items():
        piece = f"{value}"
        token_list.append(piece)

    dataset = load_from_disk(args.data_dir)
    os.makedirs(args.temp_dir, exist_ok=True)

    if not os.path.exists(f'{args.temp_dir}/midi_data.txt'):
        split = ["test", "validation", "train"]
        for split_type in split:
            for data in tqdm(dataset[split_type]):
                output_text = data['output']
                chunk_size = args.max_sentence_length  
                for i in range(0, len(output_text), chunk_size):
                    chunk = output_text[i:i+chunk_size]
                    with open(f'{args.temp_dir}/midi_data.txt', 'a') as f:
                        f.write(chunk + '\n')

    print(len(token_list))



    spm.SentencePieceTrainer.train(input=f'{args.temp_dir}/midi_data.txt', 
                                   model_prefix = args.model_prefix, 
                                   vocab_size = 20000,
                                   model_type = 'bpe',
                                   max_sentence_length = args.max_sentence_length, 
                                   max_sentencepiece_length = 128,
                                   character_coverage = 1,
                                   num_threads = 32,
                                   split_by_number = False,
                                   split_by_unicode_script = False,
                                   split_by_whitespace = False,
                                #    control_symbols= token_list[:713]
                                   )

