# fucking matplotlib
# type: ignore
import os

import matplotlib.pyplot as plt
import neptune.new as neptune
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager as fm
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler


def scatterplot(
    y_hat: np.ndarray, y_true: np.ndarray, xlabel: str, ylabel: str, fname: str, title: str, run: neptune.Run
) -> None:
    fig = plt.figure()
    plt.scatter(x=y_true, y=y_hat, alpha=0.5, s=10)
    m, b = np.polyfit(y_true, y_hat, 1)
    plt.plot(y_true, m * y_true + b, "r:", alpha=0.2, linewidth=2)
    fig.suptitle(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    run[title].log(fig)
    fig.savefig(fname)
    plt.clf()


def circumplex_model(val: np.ndarray, aro: np.ndarray, title, fname, run: neptune.Run) -> None:
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
    circle1 = plt.Circle((0.5, 0.5), 0.5, color="0.25", fill=False)
    ax.add_artist(circle1)

    ax.scatter(val, aro, color="orange", alpha=0.5, s=20)
    ax.grid(True)

    ax.set_xlabel("Valence", size=plt_size * 3, fontproperties=head_font)
    ax.set_ylabel("Arousal", size=plt_size * 3, fontproperties=head_font)
    ax.set_title(title, size=plt_size * 3, fontproperties=head_font)

    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.axhline(0, linewidth=0.1, color="black")
    ax.axvline(0, linewidth=0.1, color="black")

    # print emotion labels
    ax.text(rescale(0.98), rescale(0.35), "Happy", size=int(plt_size * 2.5), fontproperties=body_font)
    ax.text(rescale(0.5), rescale(0.9), "Excited", size=int(plt_size * 2.5), fontproperties=body_font)
    ax.text(rescale(-1.16), rescale(0.35), "Afraid", size=int(plt_size * 2.5), fontproperties=body_font)
    ax.text(rescale(-0.7), rescale(0.9), "Angry", size=int(plt_size * 2.5), fontproperties=body_font)
    ax.text(rescale(-1.13), rescale(-0.25), "Sad", size=int(plt_size * 2.5), fontproperties=body_font)
    ax.text(rescale(-0.9), rescale(-0.9), "Depressed", size=int(plt_size * 2.5), fontproperties=body_font)
    ax.text(rescale(0.98), rescale(-0.25), "Content", size=int(plt_size * 2.5), fontproperties=body_font)
    ax.text(rescale(0.7), rescale(-0.9), "Calm", size=int(plt_size * 2.5), fontproperties=body_font)

    # ax.set_aspect('equal')
    run[title].log(fig)
    plt.savefig(fname)
    plt.clf()


def conf_mat(y_pred: pd.DataFrame, y_true: pd.DataFrame, labels: list, fname: str, run: neptune.Run) -> None:
    y_true = y_true.astype(int)
    cf_mat = confusion_matrix(y_pred, y_true)
    plot = sns.heatmap(cf_mat, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels, fmt="g")
    run[fname].upload(plot.figure)
    plt.savefig(fname)
    plt.clf()


# [-1, 1] to [0, 1]
def rescale(i):
    return (0) + ((1 - (0)) / (1 - (-1))) * (i - (-1))
