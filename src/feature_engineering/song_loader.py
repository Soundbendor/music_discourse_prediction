import pandas as pd
import json
from os import walk


def _song_csv_generator(path: str):
    for subdir, _, files in walk(path):
        for file in files:
            fdir = subdir + "/" + file
            yield fdir

# WARN - THIS IS VERY BAD.
# This function truncates any files which did not have comments in the file. 
def _dejsonify(path: str) -> pd.DataFrame:
    
    with open(path) as fp:
        file = json.load(fp)
        try:
            if any([True for x in file['submissions'] if x['comments']]):
                return pd.json_normalize(file, ["submissions", "comments"],
                    meta=['song_name', 'artist_name', 'query_index', 'valence', 'arousal', 'dataset',
                    ['submission', 'title'], ['submission', 'body'], ['submission', 'lang'], ['submission', 'lang_p'],
                    ['submission', 'url'], ['submission', 'id'], ['submission', 'score'], ['submission', 'n_comments'],
                    ['submission', 'subreddit']])    
            else:
                raise IndexError
        except IndexError:
            return pd.DataFrame({
                    'song_name': file['song_name'], 
                    'artist_name':  file['artist_name'],
                    'query_index': file['query_index'],
                    'valence': file['valence'],
                    'arousal': file['arousal'],
                    'dataset': file['dataset'],
                }, index = [0])
        


def get_song_df(path: str) -> pd.DataFrame: 
    return pd.concat([_dejsonify(f) for f in _song_csv_generator(path)], axis=0, ignore_index=True)
    


               