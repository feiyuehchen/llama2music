
# midi to music (wav, mp3)
1. source of the soundfont: https://schristiancollins.com/generaluser.php
2. run 
```
cd ./scripts
python midi2music.py --midi_dir <midi_directory> --wav_dir <wav_directory> --mp3_dir <mp3_directory>
```

python midi2music.py --midi_dir ../../music_dataset/Hooktheory/dataset --wav_dir ../../music_dataset/Hooktheory/wav --mp3_dir ../../music_dataset/Hooktheory/mp3


# music to text
1. Download the Hooktheory dataset (json file): https://github.com/chrisdonahue/sheetsage#hooktheory-dataset
2. run `hooktheory2midi.py` (need to change the input directory and output directory)
3. clone the repo https://github.com/seungheondoh/lp-music-caps
4. 
