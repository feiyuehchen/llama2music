from keybert import KeyBERT
import argparse
kw_model = KeyBERT()
import json

def get_keywords(text):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,3), stop_words=None)
    return keywords


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../dataset/lyrics/clean_lyrics.json")
    
    
    
    
    args = parser.parse_args()

    