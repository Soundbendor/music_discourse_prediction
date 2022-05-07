import argparse
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import matplotlib.pyplot as plt
from nltk.corpus import stopwords as nltk_stopwords
from feature_engineering.song_loader import get_song_df
from feature_engineering.wordbag_model.wordbag import loaders, wlists

import re
import itertools
import pandas as pd
import numpy as np

wlists = {
    "eANEW": "BRM-emot-submit.csv",
    # "ANEW_Ext_Condensed": "ANEW_EnglishShortened.csv",
    "EmoLex": "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    "EmoVAD": "NRC-VAD-Lexicon.txt",
    "EmoAff": "NRC-AffectIntensity-Lexicon.txt",
    # "HSsent": "HS-unigrams.txt",
    "MPQA": "MPQA_sentiment.csv"
}


def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make pretty wordclouds - now featuring fuck-word-filtration for capstone compliance")
    parser.add_argument('-i', '--input_dir', dest='input', type=str,
                        help="Path to the directory storing the JSON files for social media data.")
    return parser.parse_args()


def _clean_str(df: pd.Series):
    rx = re.compile(r'(?:<.*?>)|(?:http\S+)')
    df = df.apply(lambda x: rx.sub('', x))
    return df

def main():
    args = parseargs()
    emotive_words = pd.concat([loaders[wlist](f"etc/wordlists/{wlists[wlist]}") for wlist in wlists.keys()])
    # words_arr = ''.join(itertools.chain(*[x.split() for x in _clean_str(get_song_df(args.input)['body'])]))
    words_arr = itertools.chain(*[x.split() for x in _clean_str(get_song_df(args.input)['body'])])
    maskArray = np.array(Image.open('mask.jpg'))
    stopwords = set(nltk_stopwords.words('english') + ['fuck', 'shit', 'dammit', 'hell'])
    words_arr = ''.join(list(pd.Series(words_arr).isin(emotive_words)))
    cloud = WordCloud(background_color='white', max_words=200, mask=maskArray, stopwords=stopwords).generate(words_arr)
    cloud.to_file('wordcloud.png')


if __name__ == '__main__':
    main()
