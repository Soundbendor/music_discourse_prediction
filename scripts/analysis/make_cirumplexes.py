import src.prediction.visualization.visualizations as vs
import pandas as pd

# load dataframes

df_deam = pd.read_csv('out/all_wordlist_features/all_DEAM_All.csv')
df_pmemo = pd.read_csv('out/all_wordlist_features/all_PmEmo_All.csv')
df_deezer = pd.concat([pd.read_csv('/mnt/d/Datasets/Deezer_2018/source/test.csv'),
                       pd.read_csv('/mnt/d/Datasets/Deezer_2018/source/train.csv'),
                       pd.read_csv('/mnt/d/Datasets/Deezer_2018/source/validation.csv')])
df_amg = pd.read_csv('out/all_wordlist_features/all_AMG1608_All.csv')

# make graphs!
vs.circumplex_model(df_deam, "DEAM Label Distribution", "tmp/circumplex/deam_noscale.png", val_key='valence', aro_key='arousal')
vs.circumplex_model(df_deezer, "Deezer Label Distribution", "tmp/circumplex/deezer_noscale.png", val_key='valence', aro_key='arousal')
vs.circumplex_model(df_pmemo, "PmEmo Label Distribution", "tmp/circumplex/pmemo_noscale.png", val_key='valence', aro_key='arousal')
vs.circumplex_model(df_amg, "AMG1608 Label Distribution", "tmp/circumplex/amg1608_noscale.png", val_key='valence', aro_key='arousal')