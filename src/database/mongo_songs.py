# Instantiation script: Add music affective datasets to song collection in mongoBD
# Expects song title, track name, and some emotion dimension (currently valence/arousal)

# A song should be a document. It should contain: 
# * global ID
# * dataset ID
# * query index (dataset specific ID)
# * artist name 
# * track title
# - emotion_label_type
# - e.g. "Dimensional (VA)"
# - emotion label
# * - e. g. (0.VA, 0.AR)
# - (*) genre
# - (*) lastFM tags
# - (*) lyrics ($lookup)
# - comments ($lookup)

# - Need to set up indexes to accelerate queries

import argparse
import os
import pymongo
import functools
import pandas as pd
from typing import List, Union 
from pymongo import MongoClient


def default_load(path: str, ds_name: str, label_type: str="Dimensional (Valence, Arousal)", dropcols: Union[None, List]=None) -> pd.DataFrame:
    df = pd.DataFrame(pd.read_csv(path))
    if dropcols:
        df.drop(dropcols, axis=1, inplace=True)
    df['Dataset'] = ds_name
    df['label_type'] = label_type
    return df


def process_deezer_subset(subset: str, path: str, label_type: str) -> pd.DataFrame:
    df = pd.DataFrame(pd.read_csv(f"{path}_{subset}.csv"))
    df['Dataset'] = "deezer"
    df['label_type'] = label_type
    df['subset'] = subset
    return(df)


def deezer_load(path: str, ds_name: str, label_type: str="Dimensional (Valence, Arousal)", dropcols: Union[None, List]=None) -> pd.DataFrame:
    deezer_apply = functools.partial(process_deezer_subset, path=path.split('_')[0], label_type=label_type)
    df = pd.concat(list(map(deezer_apply, ['test', 'train', 'validation']))) 
    print("DEBUG")
    print(df)
    return df


loaders = {
        'amg1608': functools.partial(default_load, dropcols=['Genre']),
        'deam': default_load,
        'pmemo': functools.partial(default_load, dropcols=['Genre']),
        'deezer': deezer_load
        }


def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="An ingest script for adding MER datasets to the Music Emotion Prediction dataset.")
    parser.add_argument('-i', dest='input', required=True, help='Input path for CSV')
    return parser.parse_args()


def insert_songs(path: str, collection: pymongo.collection.Collection) -> None:
    ds_name = os.path.basename(path).split('.')[0].lower()
    try:
        df = loaders[ds_name.split('_')[0]](path=path, ds_name=ds_name)
        collection.insert_many(df.to_dict('records'))
    except KeyError:
        print("Error: Unable to find dataset. Make sure you implement a load function and add it to the loaders dictionary")
        exit()


def main() -> None:
    client = MongoClient()
    db = client['mdp']
    songs = db['songs']
    args = parseargs()
    #  print( [args.input+x for x in  os.listdir(args.input)])
    if os.path.isdir(args.input):
        any(map(functools.partial(insert_songs, collection=songs), [args.input+x for x in  os.listdir(args.input)]))
    else:
        insert_songs(args.input, collection=songs)
    

if __name__ == '__main__':
    main()
