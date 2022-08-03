import os
import glob
import pandas as pd
from collections import defaultdict
from scipy.stats import pearsonr


results = defaultdict(dict)
datasets = {
    'AMG1608': ['reddit', 'youtube', 'twitter', 'All', 'MOAF'],
    'DEAM': ['reddit', 'youtube', 'twitter', 'All', 'MOAF'],
    'PmEmo': ['reddit', 'youtube', 'twitter', 'All', 'MOAF'],
    'Deezer': ['reddit', 'twitter', 'All', 'MOAF'],
}

for model in ['lr']:
    for ds in datasets.keys():
        for src in datasets[ds]:
            features = glob.glob(f"out/features/all_{ds}_{src}.csv")[0]
            os.system(f"prediction -c etc/experiments/regression_experiment_{model}.ini {features} -o {src}_all_{model}")

            predictions = pd.read_csv(f"out/predictions/{src}_all_{model}_predictions_out.csv")
            ground_truth = pd.read_csv(f"out/predictions/{src}_all_{model}_actual_out.csv")

            for key in ['valence', 'arousal']:
                results[f"{model}_{ds}_{src}"][key] = pearsonr(predictions[key], ground_truth[key])[0]
    df = pd.DataFrame(results)
    df.to_csv(f"{model}_correlation_results.csv")
    df.to_latex(f"{model}_correlation_results.tex", float_format="%.2f")
