import pandas as pd
import os

path = 'out/features'
dataset = 'DEAM'

df_reddit = pd.read_csv(os.path.join(path, f"all_{dataset}_reddit.csv"))


df_twitter = pd.read_csv(os.path.join(path, f"all_{dataset}_twitter.csv"))
df_twitter_index = df_twitter['query_index']
df_twitter = df_twitter.iloc[:,10:]
df_twitter['query_index'] = df_twitter_index


df_youtube = pd.read_csv(os.path.join(path, f"all_{dataset}_youtube.csv"))
df_youtube_index = df_youtube['query_index']
df_youtube = df_youtube.iloc[:,10:]
df_youtube['query_index'] = df_youtube_index


df2 = df_reddit.merge(df_youtube, how='inner', on='query_index', suffixes=('_reddit', '_youtube'))
df3 = df2.merge(df_twitter, how='inner', on='query_index', suffixes=('', '_twitter_'))
print(df3)

df3.to_csv(f'{dataset}_MOAF.csv')

