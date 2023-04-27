from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.tokenize import wordpunct_tokenize

from database.driver import Driver

SOURCES = ["Reddit", "Youtube", "Twitter"]
DATASET = ["deam_new", "amg1608", "pmemo"]
# Averages for all individual sources and datasets


# Cumulative histograms
def make_word_per_comment_hist(src: Union[List[str], str]) -> None:
    df = db_con.get_discourse(ds_name=src, source_type=["Reddit", "Youtube"])
    print(df["body"].apply(lambda x: len(wordpunct_tokenize(x))).describe())
    hist = sns.histplot(data=df["body"].apply(lambda x: len(wordpunct_tokenize(x))), kde=True, bins=range(0, 1025, 64))
    return hist


def make_word_hist(src: Union[List[str], str]) -> None:
    df = db_con.get_discourse(ds_name=src, source_type=["Reddit", "Youtube"])
    # Currently counts # words per comment
    # df = df.groupby(["song_name"])["body"].apply(lambda x: len(wordpunct_tokenize(x))).sum()
    df["body"] = df["body"].apply(lambda x: len(wordpunct_tokenize(x)))
    df = df.groupby(["song_name"])["body"].sum()
    print(df.describe())
    hist = sns.histplot(data=df, kde=True, bins=range(0, 1025, 64))
    return hist


def make_comment_hist(src: Union[List[str], str]) -> None:
    df = db_con.get_discourse(ds_name=src, source_type=["Reddit", "Youtube"])
    print(df["body"].apply(lambda x: len(wordpunct_tokenize(x))).describe())
    hist = sns.histplot(data=df["body"].apply(lambda x: len(wordpunct_tokenize(x))), kde=True, bins=range(0, 1025, 64))
    return hist


sns.color_palette("rocket", as_cmap=True)
db_con = Driver("mdp")
hist = None
for ds in DATASET:
    hist = make_word_hist(ds)

# Bad stupid code design
hist.set(xscale="log")
hist.set(xlabel="# Words", ylabel="Songs")
plt.legend(labels=DATASET)
fig = hist.get_figure()
fig.savefig(f"all_dist_word.png")
