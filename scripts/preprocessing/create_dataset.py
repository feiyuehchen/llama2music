import json
import argparse
import os
import glob
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split

from miditoolkit import MidiFile
from statistics import mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_dir", default="../../dataset/piano_data/raw",
        help="directory of raw data"
    )
    parser.add_argument(
        "--save_dir", default="../../dataset/piano_data/seg_midi_data",
        help="saving directory of train, validation, test midi files",
    )
    parser.add_argument(
        "--segment_length", default=None, type=int,
        help="slice the midi files into <slice> sec",
    )
    args = parser.parse_args()
    
    yaml_dict = {}
    information_json = []
    for data in os.listdir(args.raw_data_dir):
        print('='*100)
        print(data)     

        yaml_dict[data] = None
        train_save_path = os.path.join(args.save_dir, 'train', data)
        valid_save_path = os.path.join(args.save_dir, 'validation', data)
        test_save_path = os.path.join(args.save_dir, 'test', data)
        for save_path in [train_save_path, valid_save_path, test_save_path]:
            os.makedirs(save_path, exist_ok=True)
        
        # if dataset have split already
        data_split_folder = os.listdir(os.path.join(args.raw_data_dir, data))
        if 'train' in data_split_folder:
            train_list = glob.glob(f"{args.raw_data_dir}/{data}/train/**/*.mid", recursive=True) + glob.glob(f"{args.raw_data_dir}/{data}/train/**/*.midi", recursive=True)
            if 'validation' in data_split_folder and 'test' in data_split_folder:
                valid_list = glob.glob(f"{args.raw_data_dir}/{data}/validation/**/*.mid", recursive=True) + glob.glob(f"{args.raw_data_dir}/{data}/validation/**/*.midi", recursive=True)
                test_list = glob.glob(f"{args.raw_data_dir}/{data}/test/**/*.mid", recursive=True) + glob.glob(f"{args.raw_data_dir}/{data}/test/**/*.midi", recursive=True)
            else:
                valid_list = glob.glob(f"{args.raw_data_dir}/{data}/validation/**/*.mid", recursive=True) + glob.glob(f"{args.raw_data_dir}/{data}/validation/**/*.midi", recursive=True)
                test_list = []
            
        else:
            train_list = glob.glob(f"{args.raw_data_dir}/{data}/**/*.mid", recursive=True) + glob.glob(f"{args.raw_data_dir}/{data}/**/*.midi", recursive=True)
            valid_list = []
            test_list = []

        print(f"original dataset:\n train_list: {len(train_list)}")
        print(f"valid_list: {len(valid_list)}")
        print(f"test_list: {len(test_list)}")

        print("start split")

        if len(train_list) > 5000:
            train_ratio = 0.95
            validation_ratio = 0.04
            test_ratio = 0.01
        else:
            train_ratio = 0.8
            validation_ratio = 0.1
            test_ratio = 0.1    

        random_state=42
        if len(valid_list) == 0:
            train_list, valid_list = train_test_split(train_list, test_size=1 - train_ratio, random_state=random_state)
        if len(test_list) == 0:
            valid_list, test_list= train_test_split(valid_list, test_size=test_ratio/(test_ratio + validation_ratio), random_state=random_state) 

        print(f"new split dataset:\n train_list: {len(train_list)}")
        print(f"valid_list: {len(valid_list)}")
        print(f"test_list: {len(test_list)}")

        def copy_midi(data_list, midi_save_path, segment_length):
            print('='*100)
            print(f"midi_save_path: {midi_save_path}")
            print(f'Segment Length: {segment_length}')

            for midi_path in tqdm(data_list):  
                new_midi_path = os.path.join(midi_save_path, os.path.split(midi_path)[1])
                new_midi_path = new_midi_path.replace(',', '').replace(' ', '_')
                if segment_length:

                    midi_obj = MidiFile(midi_path)
                    tempo_list = [tempo.tempo for tempo in midi_obj.tempo_changes]
                    one_sec_const = midi_obj.ticks_per_beat*mean(tempo_list)/60
                    cur_start_sec, cur_end_sec = 0, 0+segment_length
                    while cur_start_sec*one_sec_const < midi_obj.max_tick:
                        seg_midi_path = new_midi_path.replace('.mid', f'_{cur_start_sec}-{cur_end_sec}.mid')
                        cst = cur_start_sec*one_sec_const
                        cet = cur_end_sec*one_sec_const
                        midi_obj.dump(seg_midi_path, segment=(cst, cet))
                        cur_start_sec, cur_end_sec = cur_start_sec+segment_length, cur_end_sec+segment_length
                    
                else:
                    os.system(f'cp "{midi_path}" "{new_midi_path}"')
            print('='*100)

        copy_midi(train_list, train_save_path, args.segment_length)
        copy_midi(valid_list, valid_save_path, args.segment_length)
        copy_midi(test_list, test_save_path, args.segment_length)


        
        information_json.append({
            "dataset_type": data,
            "save_path": args.raw_data_dir,
            "split_size": [len(train_list), len(valid_list), len(test_list)],
            "train_list": train_list,
            "valid_list": valid_list,
            "test_list": test_list
        })
        
    with open(os.path.join(args.save_dir, 'information.json'), 'w') as f:
        json.dump(information_json, f, indent=4)

    with open(os.path.join(args.save_dir, 'dataprocess.yaml'), 'w') as f:
        yaml.dump(yaml_dict, f)

    # show
    print(f'files in train set: {len(glob.glob(f"{train_save_path}/**/*.mid", recursive=True) + glob.glob(f"{train_save_path}/**/*.midi", recursive=True))}')
    print(f'files in validation set: {len(glob.glob(f"{valid_save_path}/**/*.mid", recursive=True) + glob.glob(f"{valid_save_path}=/**/*.midi", recursive=True))}')
    print(f'files in test set: {len(glob.glob(f"{test_save_path}/**/*.mid", recursive=True) + glob.glob(f"{test_save_path}/**/*.midi", recursive=True))}')