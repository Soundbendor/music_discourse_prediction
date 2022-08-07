import pandas as pd
import os
from feature_engineering.song_loader import get_song_df 


def main():
    path = '/mnt/f/fix_data/'
    dataset = 'AMG1608'
    source = 'reddit'
    df = get_song_df(os.path.join(path, dataset, source))
    print(df['query_index'].nunique())
    print(df.columns)
    df.to_csv('testing_amg_song_loader.csv')


if __name__ == '__main__':
    main()