import argparse
# import cudf
import json
import numpy as np
import pandas as pd
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from os import walk
from datetime import datetime
from typing import List



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
        return pd.json_normalize(json.load(fp), ["submissions", "comments"],
                meta=['song_name', 'artist_name', 'query_index', 'valence', 'arousal', 'dataset',
                ['submission', 'title'], ['submission', 'body'], ['submission', 'lang'], ['submission', 'lang_p'],
                ['submission', 'url'], ['submission', 'id'], ['submission', 'score'], ['submission', 'n_comments'],
                ['submission', 'subreddit']])


def _tokenize_comment(comment: str):
    rx = re.compile(r'(?:<.*?>)|(?:[^\w\s\'])|(?:\d+)')
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')  

    return pd.Series(
            filter(lambda x: x not in stop_words,
                map(lemmatizer.lemmatize,
                    nltk.word_tokenize(
                        rx.sub('', comment)
                    )
                )
            )
        ).value_counts().reset_index().rename(columns={'index': 'Word', 0: 'Count'})

def vectorize_comment(x: pd.Series, wordlist: pd.DataFrame):
    c_vec = (pd.concat(list(x), axis=0, ignore_index=True)
            .pipe(pd.merge, wordlist, on='Word')
            .drop(['Word', 'Count'], axis=1)
            .aggregate(['mean', 'min', 'max', 'std'])
            .stack()
            )

    c_vec.index = pd.Index(map(lambda x: f"{x[0]}.{x[1]}", c_vec.index.to_flat_index()))
    return c_vec.to_frame().T

def tokenize_comments(df: pd.DataFrame):
    df['body'] = df['body'].map(_tokenize_comment)
    return df

def main():
    args = parseargs()
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    # load wordlist
    wlist_path = f"etc/wordlists/{wlists[args.wordlist]}"
    # Must be done in each individual list reader, as different DFs must be loaded with different config options

    wordlist = pd.read_csv(wlist_path, names=['Word','Valence','Arousal','Dominance'], skiprows=1,  sep='\t')
    timestamp = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
    fname = f"{args.dataset}_{args.sm_type}_{timestamp}_{args.wordlist}_features.csv"

    uncompressible_cols = ['submission.subreddit', 'submission.id', 'submission.url', 'submission.lang',
                'submission.lang_p', 'id', 'lang', 'lang_p', 'replies',
                'submission.title', 'submission.body']

    # TODO - Handle submission titles, submission bodies, WITHOUT dropping them. Tokenize and emovectorize.
    df = (pd.concat([dejsonify(p) for p in song_csv_generator(args.input)], axis=0, ignore_index=True)
            .pipe(tokenize_comments)
            .drop(uncompressible_cols, axis=1))

    emo_word_stats = df.groupby(['query_index'])['body'].apply(lambda x: vectorize_comment(x, wordlist))

    df.drop('body', axis=1, inplace=True)

    df = df.groupby(['query_index']).aggregate({
        'score': 'mean',
        'submission.n_comments': lambda x: x.apply(pd.to_numeric).mean(),
        'submission.score': lambda x: x.apply(pd.to_numeric).mean(),
        'arousal': lambda x: x.iloc[0],
        'valence': lambda x: x.iloc[0],
        'dataset': lambda x: x.iloc[0],
        'artist_name': lambda x: x.iloc[0], 
        'song_name': lambda x: x.iloc[0],
    })


    df3 = df.join(emo_word_stats)
    df3.to_csv(fname)
   
