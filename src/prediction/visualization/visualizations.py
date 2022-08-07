import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix


def circumplex_model(data: pd.DataFrame, title, fname, val_key='Valence', aro_key='Arousal') -> None:
#     os.makedirs(os.path.dirname(fname), exist_ok=True)

    plt_size = 10
    fig, ax = plt.subplots(figsize=(plt_size, plt_size))
    plt.xlim(-0.12, 1.12)
    plt.ylim(-0.12, 1.12)


    header_path = "assets/Stratum2-Medium.otf"
    head_font = fm.FontProperties(fname=header_path)

    body_path = "assets/KievitOffc.ttf"
    body_font = fm.FontProperties(fname=body_path)

    # draw the unit circle
    fig = plt.gcf()
    ax = fig.gca()
    circle1 = plt.Circle((0.5, 0.5), 0.5, color='0.25', fill=False)
    ax.add_artist(circle1)

    ax.scatter(data[val_key], data[aro_key], color='orange', alpha=0.5, s=20)
    ax.grid(True)

    ax.set_xlabel("Valence", size=plt_size*3, fontproperties=head_font)
    ax.set_ylabel("Arousal", size=plt_size*3, fontproperties=head_font)
    ax.set_title(title, size=plt_size*3, fontproperties=head_font)

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.axhline(0, linewidth=0.1, color="black")
    ax.axvline(0, linewidth=0.1, color="black")

    # print emotion labels
    ax.text(rescale(0.98), rescale(0.35), 'Happy', size=int(
        plt_size*2.5), fontproperties=body_font)
    ax.text(rescale(0.5), rescale(0.9), 'Excited', size=int(
        plt_size*2.5), fontproperties=body_font)
    ax.text(rescale(-1.16), rescale(0.35), 'Afraid', size=int(plt_size*2.5),
            fontproperties=body_font)
    ax.text(rescale(-0.7), rescale(0.9), 'Angry', size=int(plt_size*2.5),
            fontproperties=body_font)
    ax.text(rescale(-1.13), rescale(-0.25), 'Sad', size=int(plt_size*2.5),
            fontproperties=body_font)
    ax.text(rescale(-0.9), rescale(-0.9), 'Depressed', size=int(plt_size*2.5),
            fontproperties=body_font)
    ax.text(rescale(0.98), rescale(-0.25), 'Content', size=int(plt_size*2.5),
            fontproperties=body_font)
    ax.text(rescale(0.7), rescale(-0.9), 'Calm', size=int(plt_size*2.5), fontproperties=body_font)

    # ax.set_aspect('equal')
    plt.savefig(fname)
    plt.clf()


def conf_mat(y_pred: pd.DataFrame, y_true: pd.DataFrame, labels: list, fname: str) -> None:
    cf_mat = confusion_matrix(y_pred, y_true)
    plot = sns.heatmap(cf_mat, annot=True, cmap='Blues',
                       xticklabels=labels, yticklabels=labels)
    plt.savefig(fname)
    plt.clf()

# [-1, 1] to [0, 1]
def rescale(i):
    return (0) + ((1 - (0)) / (1 - (-1)))*(i-(-1))