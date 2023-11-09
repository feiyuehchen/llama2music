import openai
import yaml
import argparse
import os
import json
from tqdm import tqdm
from langdetect import detect
from utils import traverse_dir



def prepare_data(args):
    print('================================================================================')
    print('PREPARE DATA...')
    wav_list = traverse_dir(args.audio_dir, is_sort=True, extension=('mp3', 'wav'))
    all_data = []
    count = 1
    for file_path in tqdm(wav_list):
        audio_tensor_list = get_audio(file_path)
        number_of_chunks = range(audio_tensor_list.shape[0])
        for chunk, audio_tensor in zip(number_of_chunks, audio_tensor_list):
            time = f"{chunk * 10}:00-{(chunk + 1) * 10}:00"

            all_data.append({
                'fname': f"[{os.path.split(file_path)[1]}]-[{time}]" ,
                'fpath': file_path,
                'audio_tensor': audio_tensor
            })
            
            # check
            if len(all_data) == 1:
                print(all_data)

        # split and save npy files because it might be too large
        if len(all_data) > 1000:
            save_path = os.path.join(args.npy_dir, str(count)+'.npy')
            print(f"save file: {save_path}")
            np.save(save_path, all_data)
            count += 1
            all_data = []

    


    
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset


def get_audio(audio_path, duration=10, target_sr=16000):
    n_samples = int(duration * target_sr)
    audio, _ = load_audio(
        path= audio_path,
        ch_format= STR_CH_FIRST,
        sample_rate= target_sr,
        downmix_to_mono= True,
    )
    if len(audio.shape) == 2:
        audio = audio.mean(0, False)  # to mono
    input_size = int(n_samples)
    if audio.shape[-1] < input_size:  # pad sequence
        pad = np.zeros(input_size)
        pad[: audio.shape[-1]] = audio
        audio = pad
    ceil = int(audio.shape[-1] // n_samples)
    audio = torch.from_numpy(np.stack(np.split(audio[:ceil * n_samples], ceil)).astype('float32'))
    return audio

class MC_Dataset(Dataset):
    def __init__(self, data_path, caption_type, sr=16000, duration=10, audio_enc="wav"):
        self.data_path = data_path
        self.caption_type = caption_type
        self.audio_enc = audio_enc
        self.n_samples = int(sr * duration)
        self.get_fl()
        
    
    def get_fl(self):
        self.fl = np.load(self.data_path, allow_pickle=True)



    def __getitem__(self, index):
        item = self.fl[index]
        fname = item['fname']
        audio_tensor = item['audio_tensor']
        
        return fname, audio_tensor

    def __len__(self):
        return len(self.fl)


def main_worker(args):
    print('================================================================================')
    print('LOAD DATASET...')
    data_list = traverse_dir(args.npy_dir, is_sort=True, extension='npy')
    
    model = BartCaptionModel(
            max_length = args.max_length,
            label_smoothing = args.label_smoothing,
        )
    lpmc_dir = f"{args.lpmc_dir}/lpmc/music_captioning/exp/{args.framework}/{args.caption_type}/"
    config = OmegaConf.load(os.path.join(lpmc_dir, "hparams.yaml"))
    model, save_epoch = load_pretrained(args, lpmc_dir, model, mdp=config.multiprocessing_distributed)
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    model.eval()
    print('================================================================================')
    print('START EVAL')
    

    for id, data_path in enumerate(data_list[args.npy_id:]):
        print(f"id: {id}, data_path: {data_path}")
        test_dataset = MC_Dataset(
            data_path =  data_path,
            caption_type = "gt"
        )
        print(f"length of the dataset: {len(test_dataset)}")
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=False)
        
        eval(args, model, test_dataset, test_loader, args.num_beams)
        print('================================================================================')
 
def eval(args, model, test_dataset, test_loader, num_beams=5):
    
    
    inference_results = {}
    if os.path.exists(args.save_path):
        f = open(args.save_path)
        inference_results = json.load(f)
    print(f'inference_results length: {len(inference_results)}')


    for batch in tqdm(test_loader):
        fname, audio_tensor = batch
        if args.gpu is not None:
            audio_tensor = audio_tensor.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            output = model.generate(
                samples=audio_tensor,
                num_beams=num_beams,
            )
        
        
        for audio_id_time, pred in zip(fname, output):
            audio_id, audio_time = audio_id_time.split(']-[')
            audio_id = audio_id.replace('[', '')
            audio_time = audio_time.replace(']', '')
            


            if audio_id not in inference_results:
                inference_results[audio_id] = {}
            
            inference_results[audio_id][audio_time] = pred

        
        
    
    with open(args.save_path, mode="w") as io:
        json.dump(inference_results, io, indent=4)


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path", default='../../music_dataset/dataset/music2text/music2text.json', 
        help="path for the output file",
    )
    parser.add_argument(
        "--lpmc_dir", default='../../music_dataset/lp-music-caps', 
        help="directory for the lpmc",
    )
    parser.add_argument(
        "--audio_dir", default='../../music_dataset/Hooktheory/wav', 
        help="directory for the input audio directory",
    )
    parser.add_argument(
        "--npy_dir", default='../../music_dataset/dataset/music/raw/hooktheory/wav', 
        help="directory for the output tensor audio data",
    )
    parser.add_argument(
        "--npy_id", default=0, 
        help="set the id to do the next one",
    )
    


    # from infer.py
    parser.add_argument('--framework', type=str, default="transfer")
    parser.add_argument("--caption_type", default="lp_music_caps", type=str)
    parser.add_argument('--arch', default='transformer')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--warmup_epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--min_lr', default=1e-9, type=float)
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=1, type=int,
                        help='GPU id to use.')
    parser.add_argument('--print_freq', default=50, type=int)
    parser.add_argument("--cos", default=True, type=bool)
    parser.add_argument("--label_smoothing", default=0.1, type=float)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--num_beams", default=5, type=int)
    parser.add_argument("--model_type", default="last", type=str)
    
    args = parser.parse_args()
    
    import sys
    sys.path.append(args.lpmc_dir)
    from lpmc.music_captioning.model.bart import BartCaptionModel
    from lpmc.utils.eval_utils import load_pretrained
    from lpmc.utils.audio_utils import load_audio, STR_CH_FIRST
    from tqdm import tqdm
    from omegaconf import OmegaConf
    
    os.makedirs(args.npy_dir, exist_ok=True)
    if len(os.listdir(args.npy_dir)) == 0:
        prepare_data(args)
    

    main_worker(args)


    
    
    