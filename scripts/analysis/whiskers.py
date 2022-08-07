import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/mnt/d/Datasets/PmEmo2019/source/PmEmoFormed.csv')
df2 = pd.concat([pd.read_csv('/mnt/d/Datasets/Deezer_2018/Source/test.csv'), pd.read_csv('/mnt/d/Datasets/Deezer_2018/Source/train.csv'), pd.read_csv('/mnt/d/Datasets/Deezer_2018/Source/validation.csv')])

fig = plt.figure()
ax = fig.add_subplot()
ax.boxplot(df2[['valence', 'arousal']], labels=['Valence', 'Arousal'])
ax.set_title('Deezer Distribution')

fig.savefig('deezer_val.png')