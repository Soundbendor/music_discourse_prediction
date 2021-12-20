import argparse
import cudf
import json
import numpy as np
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

def song_csv_generator(path: str):
    for subdir, _, files in walk(path):
        for file in files:
            fdir = subdir + "/" + file
            yield fdir


def dejsonify(path: str):
    with open(path) as fp:
        return cudf.DataFrame(
            pd.json_normalize(json.load(fp), ["submissions", "comments"],
                    meta=['song_name', 'artist_name', 'query_index', 'valence', 'arousal', 'dataset',
                    ['submission', 'title'], ['submission', 'body'], ['submission', 'lang'], ['submission', 'lang_p'],
                    ['submission', 'url'], ['submission', 'id'], ['submission', 'score'], ['submission', 'n_comments'],
                    ['submission', 'subreddit']])
        )

def main():
    args = parseargs()

    # load wordlist
    wlist_path = f"../../etc/env/wordlists/{wlists[args.wordlist]}"
    # Must be done in each individual list reader, as different DFs must be loaded with different config options

    timestamp = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
    fname = f"{args.dataset}_{args.sm_type}_{timestamp}_{args.wordlist}_features.csv"

    features = cudf.DataFrame(columns=meta_features)

    df = cudf.concat([dejsonify(p) for p in song_csv_generator(args.input)], axis=0, ignore_index=True)









