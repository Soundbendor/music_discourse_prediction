import argparse
import cudf
import json
import pandas as pd

from os import walk
from datetime import datetime

wlists = {
    "eANEW": "BRM-emot-submit.csv",
    "ANEW_Ext_Condensed": "ANEW_EnglishShortened.csv",
    "EmoLex": "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    "EmoVAD": "NRC-VAD-Lexicon.txt",
    "EmoAff": "NRC-AffectIntensity-Lexicon.txt",
    "HSsent": "HS-unigrams.txt",
    "MPQA": "MPQA_sentiment.csv"
}

meta_features = ['Song_ID', 'Song_Name', 'n_words', 'valence', 'arousal', 'n_comments']

def song_csv_generator(path: str):
    for subdir, dirs, files in walk(path):
        for file in files:
            fdir = subdir + "/" + file
            yield fdir

def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature extraction toolkit for music semantic analysis from various social media platforms.")
    parser.add_argument('-i', '--input_dir', dest='input', type=str,
        help = "Path to the directory storing the JSON files for social media data.")
    parser.add_argument('-w', '--wordlist', dest='wordlist', type=str, 
        help = 'Wordlist to generate features from. Valid options are: [EmoVAD, EmoAff, EmoLex, eANEW, MPQA]')
    parser.add_argument('--source', required=True, type=str, dest='sm_type', 
        help = "Which type of social media input is being delivered.\n\
            Valid options are [Twitter, Youtube, Reddit, Lyrics].")
    parser.add_argument('--dataset', type=str, dest='dataset', required=True,
        help = "Name of the dataset which the comments represent")
    return parser.parse_args()

def main():
    args = parseargs()

    # load wordlist
    wlist_path = f"../../etc/env/wordlists/{wlists[args.wordlist]}"
    # Must be done in each individual list reader, as different DFs must be loaded with different config options
    # wlist = cudf.read_csv(wlist_path)
    timestamp = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
    fname = f"{args.dataset}_{args.sm_type}_{timestamp}_{args.wordlist}_features.csv"

    features = cudf.DataFrame(columns=meta_features)

    for idx, file in enumerate(song_csv_generator(args.input)):
        # SONGS ARE IN JSON
        # faster to load all data at once, or process file by file? 
        with open(file) as fp:
            song = json.load(fp)
            submissions = cudf.DataFrame(song['submissions'])
            # automatically a series of json objects
            comments_series = submissions['comments']
            sub2 = pd.json_normalize(song, "submissions", max_level=2,
                meta=['song_name', 'artist_name', 'query_index', 'valence', 'arousal'])
            print(sub2)
            print(sub2.columns)
            




