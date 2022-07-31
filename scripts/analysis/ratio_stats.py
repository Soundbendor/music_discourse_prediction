import pandas as pd
import os
import numpy as np
from feature_engineering.song_loader import get_song_df


path = '/mnt/f/fix_data/'
datasets = ['AMG1608', 'DEAM', 'PmEmo']
# datasets = ['Deezer']
sources = ['reddit', 'youtube', 'twitter']
# sources = ['twitter']
datasets = {
    'AMG1608': ['reddit', 'youtube', 'twitter'],
    'DEAM': ['reddit', 'youtube', 'twitter'],
    'PmEmo': ['reddit', 'youtube', 'twitter'],
    'Deezer': ['reddit', 'twitter'],
}
df_total_results = pd.DataFrame()
df_avg_results = pd.DataFrame()
df_ratio_results = pd.DataFrame()

for ds in datasets.keys():
    for src in datasets[ds]:
        df = get_song_df(os.path.join(path, ds, src))
        print(df)
        print(f"Stats for {src} in {ds}\n\n")

        df_total_results.loc[f'{ds}_{src}', 'Total Songs'] = len(
            df['query_index'].unique())
        df_total_results.loc[f'{ds}_{src}', 'Total Submissions'] = len(
            df['submission.id'].unique())
        df_total_results.loc[f'{ds}_{src}', 'Total Comments'] = len(df)

        df[['query_index', 'body']].groupby(
            'query_index').agg({'body': lambda x: sum(x.apply(lambda y: print(y)))})

        df_ratio_results.loc[f'{ds}_{src}', 'Submissions per Song'] = df[[
            'query_index', 'submission.id']].groupby('query_index').agg('nunique').mean()['submission.id']
        df_ratio_results.loc[f'{ds}_{src}', 'Comments per Song'] = df[[
            'query_index', 'submission.id']].groupby('query_index').agg('count').mean()['submission.id']
        df_ratio_results.loc[f'{ds}_{src}', 'Words per Song'] = df[['query_index', 'body']].groupby(
            'query_index').agg({'body': lambda x: sum(x.apply(lambda y: len(y.split())))}).mean()['body']
        df_ratio_results.loc[f'{ds}_{src}', 'Comments per Submission'] = df[[
            'submission.id', 'body']].groupby('submission.id').agg('count').mean()['body']
        df_ratio_results.loc[f'{ds}_{src}', 'Words per Submission'] = df[['submission.id', 'body']].groupby(
            'submission.id').agg({'body': lambda x: sum(x.apply(lambda y: len(y.split())))}).mean()['body']

        df_avg_results.loc[f'{ds}_{src}', 'Comment Avg.'] = len(
            df) / len(df['query_index'].unique())
        df_avg_results.loc[f'{ds}_{src}', 'Comment Std.'] = df[[
            'query_index', 'body']].groupby('query_index').agg('count').std()['body']
        df_avg_results.loc[f'{ds}_{src}', 'Word Avg.'] = sum(
            [len(x.split()) for x in df['body']]) / len(df['query_index'].unique())
        df_avg_results.loc[f'{ds}_{src}', 'Word Std.'] = np.std(
            [len(x.split()) for x in df['body']])

print(df_total_results)
print(df_ratio_results)
print(df_avg_results)

df_total_results.T.to_latex('out/tables/ds_summary_totals.tex', float_format="%.2f")
df_ratio_results.T.to_latex('out/tables/ds_summary_ratios.tex', float_format="%.2f")
df_avg_results.T.to_latex('out/tables/ds_summary_avg.tex', float_format="%.2f")
