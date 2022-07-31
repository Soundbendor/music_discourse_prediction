import pandas as pd
import json
import os
from tqdm import tqdm

def _song_csv_generator(path: str):
    for subdir, _, files in os.walk(path):
        for file in files:
            fdir = subdir + "/" + file
            yield fdir


def main():
    path = '/mnt/f/last_ditch_effort/'
    dataset = 'PmEmo'
    source = 'reddit'

    # df = pd.concat([pd.read_csv('/mnt/d/Datasets/Deezer_2018/Source/test.csv'), pd.read_csv('/mnt/d/Datasets/Deezer_2018/Source/train.csv'), pd.read_csv('/mnt/d/Datasets/Deezer_2018/Source/validation.csv')])
    # df = pd.read_csv('/mnt/d/Datasets/DEAM2016/source/deamformed.csv')
    df = pd.read_csv('/mnt/d/Datasets/PmEmo2019/source/PmEmoFormed.csv')

    for file in tqdm(_song_csv_generator(os.path.join(path, dataset, source))):
        with open(file, 'r') as fp:
            song = json.load(fp)
        song['query_index'] = int(df[(df['song_name'] == song['song_name']) & (df['artist_name'] == song['artist_name'])]['song_id'].values[0])
        newpath = os.path.join('/mnt/f/fix_data', dataset, source)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        with open(os.path.join(newpath, os.path.basename(file)), 'w', encoding='utf-8') as f:
            json.dump(song, f, ensure_ascii=True)

if __name__ == '__main__':
    main()