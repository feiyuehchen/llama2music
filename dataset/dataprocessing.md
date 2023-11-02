# data source

## music
1. Hooktheory dataset (json file): https://github.com/chrisdonahue/sheetsage#hooktheory-dataset
Download the Hooktheory dataset (json file): https://github.com/chrisdonahue/sheetsage#hooktheory-dataset
run `hooktheory2midi.py` (need to change the input directory and output directory)

## lyrics
1. azlyrics-scraper: https://github.com/AlbertSuarez/azlyrics-scraper
2. metrolyrics: https://github.com/ludovicaschaerf/TMCI_Project/tree/master


## others
1. source of the soundfont: https://schristiancollins.com/generaluser.php


# midi to music (wav, mp3)
1.  run 
```
cd ./scripts
python midi2music.py --midi_dir <midi_directory> --wav_dir <wav_directory> --mp3_dir <mp3_directory>
```

whole pieces
python midi2music.py --midi_dir ../../music_dataset/Hooktheory/dataset --wav_dir ../../music_dataset/Hooktheory/wav --mp3_dir ../../music_dataset/Hooktheory/mp3

melody only
python midi2music.py --midi_dir ../../music_dataset/Hooktheory/melody_only/dataset --wav_dir ../../music_dataset/Hooktheory/melody_only/wav --mp3_dir ../../music_dataset/Hooktheory/melody_only/mp3


# music to text
main idea: use lpmc to do music captioning

1. clone the repo https://github.com/seungheondoh/lp-music-caps
2. data: Hooktheory(wav file)

q: wav or mp3?


# text to music
main idea: 




# lyrics to text
main idea: use GPT3.5 turbo to get sentiment, main idea...etc

1. data: az, metro
2. instruction:
"""Please answer the following questions, with each question being answered around 100 words:
    1. Analyze the lyrics and determine the overall sentiment (e.g., happy, sad, angry) of the song. Provide a summary of the emotional tone.
    2. Identify the tone (e.g., humorous, melancholic, rebellious) and mood (e.g., relaxed, upbeat, reflective) of the song.
    3. Pick out and summarize the most iconic or memorable lines from the song.
    4. Please analyze the lyrics and classify them into one of the four categories based on Russell's Circumplex model of affect: (1) High Positive Affect, (2) High Negative Affect, (3) Low Positive Affect, or (4) Low Negative Affect. 
"""
3. prompt:
f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}

### Input:
{lyrics}
""" 

4. exc details: 1000hrs, 750$
exc code:
python lyrics2text.py --clean_path /home/feiyuehchen/personality/llama2music/dataset/lyrics/clean_lyrics.json --previos_json /home/feiyuehchen/personality/llama2music/dataset/lyrics2text/lyrics2text.json

# text to lyrics
main idea: there are several condition texts. 
a. use keybert to get keywords of the whole lyrics
b. 




# lyrics to music

# music to lyrics
"""
SOTA:
https://aclanthology.org/2023.acl-long.513.pdf

baseline:
https://arxiv.org/pdf/2012.05168.pdf
GPT-4

evaluation:
https://arxiv.org/pdf/2310.00863.pdf
https://aclanthology.org/2023.acl-long.513.pdf

"""



