import os

def main(args):
    mp3_list = os.listdir(args.MSD_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--MSD_dir", default='../../music_dataset/MSD/mp3', 
        help="directory for the mp3 files of Million Song Dataset.",
    )
    

    
    args = parser.parse_args()
    main(args)