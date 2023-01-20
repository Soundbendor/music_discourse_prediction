import argparse
import itertools
import re
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer, word_tokenize
from PIL import Image
from song_loader import get_song_df
from wordcloud import STOPWORDS, WordCloud


def _tokenize_comment(comment: str):
    rx = re.compile(r"(?:\[\S*\])|(?:<.*?>)|(?:[^\w\s\'])|(?:\d+)|(?:http\S*)")
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")

    return (
        pd.Series(
            filter(
                lambda x: x not in stop_words,
                map(
                    lemmatizer.lemmatize,
                    filter(lambda x: x not in stop_words, WhitespaceTokenizer().tokenize(text=rx.sub("", comment))),
                ),
            )
        )
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Word", 0: "Count"})
    )


def tokenize_comments(df: pd.DataFrame):
    df["body"] = df["body"].map(_tokenize_comment)
    return df


def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", dest="path", help="Path to song directory", required=True)
    return parser.parse_args()


def main() -> None:
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

    stop_words = stopwords.words("english")
    print(stop_words)

    mask = np.array(Image.open("mask.jpg"))

    args = parseargs()
    df = get_song_df(args.path).dropna(subset=["body"])
    print(df["body"])

    # corpus = list(itertools.chain.from_iterable([word_tokenize(str(x).lower()) for x in df["body"]]))
    corpus = _tokenize_comment(" ".join(df["body"]).lower())
    corpus["Word"] = corpus["Word"].astype("string")
    corpus = dict(zip(corpus["Word"], corpus["Count"]))

    wc = WordCloud(max_words=100, background_color="white", mask=mask, stopwords=STOPWORDS).generate_from_frequencies(
        corpus
    )
    plt.imshow(wc.recolor(colormap="magma"))
    plt.show()
    svg = wc.to_svg()
    with open("wordcloud.svg", "w") as f:
        f.write(svg)


if __name__ == "__main__":
    main()
