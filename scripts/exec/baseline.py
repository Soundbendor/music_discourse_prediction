import os
import glob
import pandas as pd
from collections import defaultdict
from scipy.stats import pearsonr


big_results = defaultdict(dict)
results = defaultdict(dict)
datasets = {
    'AMG1608': ['reddit', 'youtube', 'twitter', 'MOAF'],
    'DEAM': ['reddit', 'youtube', 'twitter',  'MOAF'],
    'PmEmo': ['reddit', 'youtube', 'twitter', 'MOAF'],
    'Deezer': ['reddit', 'twitter', 'MOAF'],
}

for model in ['ada', 'knn', 'lightgbm', 'rf', 'svm']:
    for ds in datasets.keys():
        for src in datasets[ds]:
            features = glob.glob(f"out/features/all_{ds}_{src}.csv")[0]
            os.system(f"prediction -c etc/experiments/regression_experiment_{model}.ini {features} -o {ds}_{src}_all_{model}")

            predictions = pd.read_csv(f"out/predictions/{ds}_{src}_all_{model}_predictions_out.csv")
            ground_truth = pd.read_csv(f"out/predictions/{ds}_{src}_all_{model}_actual_out.csv")

            for key in ['valence', 'arousal']:
                results[f"{model}_{ds}_{src}"][key] = pearsonr(predictions[key], ground_truth[key])[0]
            mdsrc = f"{model}_{ds}_{src}"
            big_results[f"{model}"][f"{ds}_{src}"] = f"({results[mdsrc]['valence']:0.3f}, {results[mdsrc]['arousal']:0.3f})"
    df = pd.DataFrame(results)
    df.to_csv(f"{model}_correlation_results.csv")
    df.to_latex(f"{model}_correlation_results.tex", float_format="%.2f")
df = pd.DataFrame(big_results)
df.to_csv("correlation_results.csv")
df.to_latex("correlation_results.tex", float_format="%.2f")