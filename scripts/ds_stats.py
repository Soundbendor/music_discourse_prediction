import argparse

import pandas as pd
from nltk.tokenize import wordpunct_tokenize


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, required=True)
    return parser.parse_args()


def main():
    args = parseargs()
    df = pd.read_csv(args.input, lineterminator="\n")

    song_groups = df.groupby(["song_name"])
    comment_counts = song_groups.size()
    print(f"Total Comments: {len(df)}")
    print(f"Mean Comments per song: {comment_counts.mean()}")
    print(f"Stdev Comments per song: {comment_counts.std()}")

    # Bad approach
    df["wordlen"] = df["body"].astype(str).apply(lambda x: len(wordpunct_tokenize(x)))
    wordcounts = df.groupby(["song_name"])["wordlen"].sum()
    print(f"Mean Words per song: {wordcounts.mean()}")
    print(f"Stdev Words per song: {wordcounts.std()}")


if __name__ == "__main__":
    main()
