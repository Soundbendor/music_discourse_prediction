import pandas as pd
import argparse
from feature_engineering.song_loader import get_song_df


# def get_songs(args: argparse.Namespace):
#     if args.intersection_type == 'intersect':
#         reddit = get_song_df(f"{args.input}/reddit")
#         twitter = get_song_df(f"{args.input}/twitter")
#         youtube = get_song_df(f"{args.input}/youtube")
#         df2 = reddit.merge(twitter, how='inner',
#                            on='query_index', suffixes=('_reddit', '_youtube'))
#         df3 = df2.merge(youtube, how='inner', on='query_index',
#                         suffixes=('', '_youtube'))
#         df = pd.concat([reddit, twitter, youtube])
#         print(df)
#         print(df3.columns)
#         df = df.loc[df[df['query_index'].isin(df3['query_index'])].index]
#         print(df.describe())
#         return df
#     return get_song_df(args.input)


input = '/mnt/f/fix_data/AMG1608'

reddit = get_song_df(f"{input}/reddit").dropna(how='any', subset=['body'])
twitter = get_song_df(f"{input}/twitter").dropna(how='any', subset=['body'])
youtube = get_song_df(f"{input}/youtube").dropna(how='any', subset=['body'])
# df2 = reddit.merge(twitter, how='inner',
#                     on='query_index', suffixes=('_reddit', '_youtube'))
# df3 = df2.merge(youtube, how='inner', on='query_index',
#                 suffixes=('', '_youtube'))
df = pd.concat([reddit, twitter, youtube])
print(df)
print(twitter['query_index'].unique())
print((df['query_index'].isin(twitter['query_index'])))
df = df[(df['query_index'].isin(twitter['query_index'])) & (df['query_index'].isin(reddit['query_index'])) & (df['query_index'].isin(youtube['query_index']))]
print(len(df['query_index'].unique()))

# print(df3.columns)
# df = df.loc[df[df['query_index'].isin(df3['query_index'])].index]
# print(df.describe())