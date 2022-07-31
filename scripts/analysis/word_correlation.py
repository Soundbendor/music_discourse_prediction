import argparse
import re
import pandas as pd

from collections import defaultdict
from os import listdir
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr


def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find correlation of average VAD words with actual song VAD labels")
    parser.add_argument('-i', dest='input', type=str, help='Pre-computed affective features')
    return parser.parse_args()


def main():
    args = parseargs()
    results = defaultdict(dict)

    for fname in listdir(args.input):
        with open(f"{args.input}/{fname}") as file:
            df = pd.read_csv(file).dropna()
            scaler = MinMaxScaler(feature_range=(0, 1))

            # Regex magic to extract social media source type from input filename
            rx = re.compile(r'[a-z]*_[a-zA-Z0-9]*_(\w*)')
            m = rx.search(fname)
            sm_source = m.group(1)

            for key in ['Valence', 'Arousal']:
                for label_key in zip(['eANEW', 'emoVAD'], [f"mean.{key[:1].upper()}.Mean.Sum", f"mean.{key}"]):
                    y_true = scaler.fit_transform(df[key.lower()].values.reshape(-1, 1)).ravel()
                    y_pred = scaler.fit_transform(df[label_key[1]].values.reshape(-1, 1)).ravel()
                    results[f"{df['dataset'].iloc[0]}_{sm_source}"][f"{key}_pearson"] = pearsonr(y_true, y_pred)[0]
                    results[f"{df['dataset'].iloc[0]}_{sm_source}"][f"{key}_r2"] = r2_score(y_true, y_pred)
                    results[f"{df['dataset'].iloc[0]}_{sm_source}"][f"{key}_mse"] = mean_squared_error(y_true, y_pred)
                    pearsons = pearsonr(y_true, y_pred)[0]
                    r2 = r2_score(y_true, y_pred)
                    mse = mean_squared_error(y_true, y_pred)
                    print(f"Summary statistics for {key} correlations to {label_key[0]} affective wordlist\n")
                    print(f"Pearsons Correlation: {pearsons}\nR2: {r2}\nMSE: {mse}\n\n")
    ds = pd.DataFrame(results)
    ds.to_csv('word_correlations.csv')

if __name__ == '__main__':
    main()
