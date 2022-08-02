from doctest import master
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import seaborn as sns



path = '/mnt/f/fix_data/'
datasets = {
    'AMG1608': ['reddit', 'youtube', 'twitter'],
    'DEAM': ['reddit', 'youtube', 'twitter'],
    'PmEmo': ['reddit', 'youtube', 'twitter'],
    'Deezer': ['reddit', 'twitter'],
}

mpl.style.use("ggplot")

for ds in datasets.keys():
    master_n_comments = pd.Series()
    master_n_words = pd.Series()
    for src in datasets[ds]:
        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
        df = pd.read_csv(f'out/features/all_{ds}_{src}.csv')

        fig, axes = plt.subplots(1, 2, constrained_layout = True)
        fig.suptitle(f'Music Discourse Distribution - {ds} - {src}')
        df['n_comments'] = df['n_comments'].replace(0, 1)
        df['n_words'] = df['n_words'].replace(0, 1)

        sns.histplot(df['n_comments'], log_scale=10, kde=True, ax=axes[0], color='orange')
        sns.histplot(df['n_words'], log_scale=10, kde=True, ax=axes[1], color='orange')
        plt.savefig(f'out/visuals/hist_{ds}_{src}.png')

        master_n_comments = master_n_comments.append(df['n_comments'])
        master_n_words = master_n_words.append(df['n_words'])

    fig, axes = plt.subplots(1, 2, constrained_layout = True)
    fig.suptitle(f'Music Discourse Distribution - {ds} - All')
    sns.histplot(master_n_comments, log_scale=10, kde=True, ax=axes[0], color='orange')
    sns.histplot(master_n_words, log_scale=10, kde=True, ax=axes[1], color='orange')
    plt.savefig(f'out/visuals/hist_{ds}_all.png')