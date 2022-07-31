import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt




wlists = {
    "eANEW": "BRM-emot-submit.csv",
    # "ANEW_Ext_Condensed": "ANEW_EnglishShortened.csv",
    "EmoLex": "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    "EmoVAD": "NRC-VAD-Lexicon.txt",
    "EmoAff": "NRC-AffectIntensity-Lexicon.txt",
    # "HSsent": "HS-unigrams.txt",
    "MPQA": "MPQA_sentiment.csv"
}

def check_affect(sub_df: pd.DataFrame, key: str):
    affects = sub_df['Affect'].reset_index(
        drop=True).str.contains(key, regex=False)
    affects = affects[affects]
    if affects.index.empty:
        return 0
    return sub_df['Score'].iloc[affects.index[0]]

def get_scores(sub_df: pd.DataFrame, df2: pd.DataFrame):
    return pd.DataFrame({
        'Word': sub_df['Word'].iloc[0],
        'Anger': check_affect(sub_df, 'anger'),
        'Joy': check_affect(sub_df, 'joy'),
        'Sadness': check_affect(sub_df, 'sadness'),
        'Fear': check_affect(sub_df, 'fear')}, index=[0])

def load_emolex(path: str) -> pd.DataFrame:
    return (pd.read_csv(path, names=['Word', 'Emotion', 'Association'], skiprows=1, sep='\t')
            .groupby(['Word']).apply(lambda x: x.set_index('Emotion')['Association'])
            .reset_index()
            )

def load_emoaff(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, names=['Word', 'Score',
                     'Affect'], skiprows=1, sep='\t')
    df2 = pd.DataFrame(columns=['Word', 'Anger', 'Joy', 'Sadness', 'Fear'])
    rows = df.groupby('Word').apply(lambda x: get_scores(x, df2))
    return df2.append(rows, ignore_index=True)

def load_mpqa(path: str) -> pd.DataFrame:
    return(pd.read_csv(path,  names=['Word', 'Sentiment'], skiprows=0)
           .drop_duplicates(subset='Word')
           .replace({'Sentiment': {'positive': 1, 'negative': -1, 'neutral': 0, 'both': 0}}))


loaders = {
    "eANEW": lambda x: pd.read_csv(x, encoding='utf-8', engine='python', index_col=0),
    "EmoLex": load_emolex,
    "EmoVAD": lambda x: pd.read_csv(x, names=['Word', 'Valence', 'Arousal', 'Dominance'], skiprows=1,  sep='\t'),
    "EmoAff": load_emoaff,
    "HSsent": None,
    "MPQA": load_mpqa,
}

df_eanew = loaders['eANEW'](f'etc/wordlists/{wlists["eANEW"]}')[['V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']]
df_emovad = loaders['EmoVAD'](f'etc/wordlists/{wlists["EmoVAD"]}')
df_emolex = loaders['EmoLex'](f'etc/wordlists/{wlists["EmoLex"]}')
df_emoaff = loaders['EmoAff'](f'etc/wordlists/{wlists["EmoAff"]}')
df_mpqa = loaders['MPQA'](f'etc/wordlists/{wlists["MPQA"]}')

print(df_eanew)
print(df_emovad)
print(df_emolex)
print(df_emoaff)
print(df_mpqa)

fig = plt.figure()
ax = fig.add_subplot()
ax.boxplot(df_emovad[['Valence', 'Arousal', 'Dominance']], labels=['Valence', 'Arousal', 'Dominance'])
ax.set_title('EmoVAD Distribution')

fig.savefig('EmoVAD_whisker.png')

fig = plt.figure()
ax = fig.add_subplot()
ax.boxplot(df_eanew[['V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']], labels=['Valence', 'Arousal', 'Dominance'])
ax.set_title('eANEW Distribution')

fig.savefig('eANEW_whisker.png')

print("\n\nStats for EmoVAD")
for dim in ['Valence', 'Arousal', 'Dominance']:
    print(f"{dim} mean: {df_emovad[dim].mean():0.3f}")
    print(f"{dim} std: {df_emovad[dim].std():0.3f}")

print("\n\nStats for eANEW")
for dim in ['V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']:
    print(f"{dim} mean: {df_eanew[dim].mean():0.3f}")
    print(f"{dim} std: {df_eanew[dim].std():0.3f}")

print(df_emovad.describe())
print(df_eanew.describe())
print(df_emolex.describe())

for name, col in df_emolex.drop('Word', axis=1).iteritems():
    print(col.value_counts())

for affect in ['Anger', 'Joy', 'Sadness', 'Fear']:
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.boxplot([df_emoaff[affect], df_emoaff[df_emoaff[affect] != 0][affect]], labels=[affect, f"{affect}_scale"])
    ax.set_title(f'emoAff Distribution - {affect}')

    fig.savefig(f'emoAff_whisker_{affect}.png')

print(df_emoaff.describe())


print(f"\n\nMPQA: Positive Count - {len(df_mpqa[df_mpqa['Sentiment'] == 1])}")
print(f"\n\nMPQA: Neutral Count - {len(df_mpqa[df_mpqa['Sentiment'] == 0])}")
print(f"\n\nMPQA: Negative Count - {len(df_mpqa[df_mpqa['Sentiment'] == -1])}")
