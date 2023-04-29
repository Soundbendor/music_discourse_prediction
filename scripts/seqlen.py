from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.tokenize import wordpunct_tokenize

from database.driver import Driver

SOURCES = ["Reddit", "Youtube", "Twitter"]
DATASET = ["deezer", "deam_new", "amg1608", "pmemo"]
STAGE_NAME = ["Deezer", "DEAM", "AMG1608", "PMEmo"]
# Averages for all individual sources and datasets
DATASET = ["deam_new", "amg1608", "pmemo"]
STAGE_NAME = ["DEAM", "AMG1608", "PMEmo"]


def get_n_words(df: pd.DataFrame) -> pd.Series:
    df["body"] = df["body"].astype(str).apply(lambda x: len(wordpunct_tokenize(x)))
    return df.groupby(["song_name"])["body"].sum()


def get_n_comments(df: pd.DataFrame) -> pd.Series:
    return df.groupby(["song_name"])["body"].size()


def make_hist(df: pd.DataFrame) -> None:
    sns.set(font_scale=1.25)
    ax = sns.histplot(data=df, x="value", hue="name", kde=True, bins=32)
    ax.set_xlim(0, 500)
    return ax


def get_song_word_ratio(df: pd.DataFrame) -> pd.Series:
    df["body"] = (
        df["body"]
        .replace("", np.nan)
        .dropna()
        .reset_index(drop=True)
        .astype(str)
        .apply(lambda x: len(wordpunct_tokenize(x)))
    )
    return df["body"]
    # return df.groupby(["song_name"])["body"].apply(lambda x: x.sum() / len(x))


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
                    get_song_word_ratio,
                    [db_con.get_discourse(ds_name=ds, source_type=["Reddit", "Youtube"]) for ds in DATASET],
                ),
                STAGE_NAME,
            ),
        )
    ),
)

df = df.replace(0, np.nan)
df = df.dropna()
df = df[np.isfinite(df["value"])]
hist = make_hist(df)

hist.set(xlabel="# Words", ylabel="Comments")
fig = hist.get_figure()
fig.savefig(f"word_comment_ratio.png")
