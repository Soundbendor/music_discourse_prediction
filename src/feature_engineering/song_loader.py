import pandas as pd
import json
from os import walk


def _song_csv_generator(path: str):
    for subdir, _, files in walk(path):
        for file in files:
            fdir = subdir + "/" + file
            yield fdir


def _dejsonify(path: str) -> pd.DataFrame:
    with open(path) as fp:
        return pd.json_normalize(json.load(fp), ["submissions", "comments"],
                meta=['song_name', 'artist_name', 'query_index', 'valence', 'arousal', 'dataset',
                ['submission', 'title'], ['submission', 'body'], ['submission', 'lang'], ['submission', 'lang_p'],
                ['submission', 'url'], ['submission', 'id'], ['submission', 'score'], ['submission', 'n_comments'],
                ['submission', 'subreddit']])


def get_song_df(path: str) -> pd.DataFrame: 
    return pd.concat([_dejsonify(f) for f in _song_csv_generator(path)], axis=0, ignore_index=True)
