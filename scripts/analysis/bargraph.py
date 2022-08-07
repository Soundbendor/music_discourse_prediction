import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

datasets = {
    'AMG1608': ['reddit', 'youtube', 'twitter', 'MOAF'],
    'DEAM': ['reddit', 'youtube', 'twitter', 'MOAF'],
    'PmEmo': ['reddit', 'youtube', 'twitter', 'MOAF'],
    'Deezer': ['reddit', 'twitter', 'MOAF'],
}

models = ['ada', 'knn', 'lightgbm', 'rf', 'svm']

def get_records(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index('Unnamed: 0').applymap(
        lambda x: [float(y) for y in x.strip('()').split(',')])
    df_records = pd.DataFrame(
        columns=['Dataset', 'Source', 'Model', 'Valence', 'Arousal'])
    for ds in datasets.keys():
        for src in datasets[ds]:
            for model in models:
                df_records = df_records.append({'Dataset': ds,
                                                'Source': src,
                                                'Model': model,
                                                'Valence': df.loc[f"{ds}_{src}"][model][0],
                                                'Arousal': df.loc[f"{ds}_{src}"][model][1]}, ignore_index=True)
    print(df_records)
    return df_records


def main():
    mpl.style.use("ggplot")
    plt.rcParams.update({'font.size': 18})
    df = pd.read_csv('out/correlation_results/correlation_results.csv')
    df_records = get_records(df)

    df_records = df_records.replace('MOAF', 'All')

    g = sns.catplot(data=df_records, hue='Model', x='Source', y='Valence', col='Dataset', kind='bar', col_wrap=2)
    g.set(ylim=(0, 1))
    plt.savefig('ml_results_valence.png')

    plt.close()
    g = sns.catplot(data=df_records, hue='Model', x='Source', y='Arousal', col='Dataset', kind='bar', col_wrap=2)
    g.set(ylim=(0, 1))
    plt.savefig('ml_results_arousal.png')


if __name__ == '__main__':
    main()
