from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.tokenize import wordpunct_tokenize

from database.driver import Driver

SOURCES = ["Reddit", "Youtube", "Twitter"]
DATASET = ["deezer", "deam_new", "amg1608", "pmemo"]
STAGE_NAME = ["Deezer", "DEAM", "AMG1608", "PMEmo"]
# Averages for all individual sources and datasets


# Cumulative histograms
def make_word_per_comment_hist(src: Union[List[str], str]) -> None:
    df = db_con.get_discourse(ds_name=src, source_type=["Reddit", "Youtube"])
    print(df["body"].apply(lambda x: len(wordpunct_tokenize(x))).describe())
    hist = sns.histplot(data=df["body"].apply(lambda x: len(wordpunct_tokenize(x))), kde=True)
    return hist


def make_word_hist(src: Union[List[str], str]) -> None:
    df = db_con.get_discourse(ds_name=src, source_type=["Reddit", "Youtube"])
    # Currently counts # words per comment
    # df = df.groupby(["song_name"])["body"].apply(lambda x: len(wordpunct_tokenize(x))).sum()
    df["body"] = df["body"].astype(str).apply(lambda x: len(wordpunct_tokenize(x)))
    df = df.groupby(["song_name"])["body"].sum()
    print(df.describe())
    hist = sns.histplot(data=df, kde=True, bins=32, log_scale=True)
    return hist


def make_comment_hist(src: Union[List[str], str]) -> None:
    df = db_con.get_discourse(ds_name=src, source_type=["Reddit", "Youtube"])
    print(df.describe())
    df = df.groupby(["song_name"])["body"].sum()
    hist = sns.histplot(data=df, kde=True)
    return hist


def get_n_words(df: pd.DataFrame) -> pd.Series:
    df["body"] = df["body"].astype(str).apply(lambda x: len(wordpunct_tokenize(x)))
    return df.groupby(["song_name"])["body"].sum()


def make_hist(df: pd.DataFrame) -> None:
    return sns.histplot(data=df, x="value", hue="name", kde=True, bins=32, log_scale=True)


sns.color_palette("rocket", as_cmap=True)
db_con = Driver("mdp")

df = pd.concat(
    axis=0,
    ignore_index=True,
    objs=list(
        map(
            lambda x: pd.DataFrame.from_dict({"value": x[0], "name": x[1]}),
            zip(
                map(
                    get_n_words, [db_con.get_discourse(ds_name=ds, source_type=["Reddit", "Youtube"]) for ds in DATASET]
                ),
                STAGE_NAME,
            ),
        )
    ),
)
hist = make_hist(df)

# Bad stupid code design
# hist.set(xscale="log")
hist.set(xlabel="# Words", ylabel="Songs")
plt.legend(labels=STAGE_NAME)
fig = hist.get_figure()
fig.savefig(f"all_dist_word.png")
