import pandas as pd
from sklearn.preprocessing import MinMaxScaler


df_deam = pd.read_csv('/mnt/d/Datasets/Deam2016/source/deamformed.csv')
df_pmemo = pd.read_csv('/mnt/d/Datasets/PmEmo2019/source/PmEmoFormed.csv')
df_deezer = pd.concat([pd.read_csv('/mnt/d/Datasets/Deezer_2018/source/test.csv'),
                       pd.read_csv('/mnt/d/Datasets/Deezer_2018/source/train.csv'),
                       pd.read_csv('/mnt/d/Datasets/Deezer_2018/source/validation.csv')])
df_amg = pd.read_csv('/mnt/d/Datasets/AMG1608/AMG1608_formed.csv')

dfs = [df_deam, df_pmemo, df_deezer, df_amg]
df_names = ['DEAM', 'PmEmo', 'Deezer', 'AMG1608']

for df, df_name in zip(dfs, df_names): 
    valence_actual_mean = df['valence'].mean()
    valence_actual_std = df['valence'].std()
    arousal_actual_mean = df['arousal'].mean()
    arousal_actual_std = df['arousal'].std()

    print(f"\n\nSummary Statistics for {df_name}")
    print(f"Valence Mean (Actual): \t {valence_actual_mean:0.3f}")
    print(f"Valence Std (Actual): \t {valence_actual_std:0.3f}")
    print(f"Arousal Mean (Actual): \t {arousal_actual_mean:0.3f}")
    print(f"Arousal Std (Actual): \t {arousal_actual_std:0.3f}")

    scaler = MinMaxScaler(feature_range=(-1,1))
    df_v_scaled = scaler.fit_transform(df['valence'].values.reshape(-1, 1))
    df_a_scaled = scaler.fit_transform(df['arousal'].values.reshape(-1, 1))

    print(f"\n\nScaled Valence Mean: {df_v_scaled.mean():0.3f}")
    print(f"Scaled Valence Deviation: {df_v_scaled.std():0.3f}")
    print(f"Scaled Arousal Mean: {df_a_scaled.mean():0.3f}")
    print(f"Scaled Arousal Deviation: {df_a_scaled.std():0.3f}")
