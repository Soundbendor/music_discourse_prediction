'''
Tool for generating a circumplex model
Graphs valence and arousal on a 0..1 range
See here for more information: https://en.wikipedia.org/wiki/Emotion_classification
~ Aidan B.
'''
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.preprocessing import MinMaxScaler

def circumplex_model(data: pd.DataFrame, title, fname, val_key='Valence', aro_key='Arousal') -> None:
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = data[[val_key, aro_key]]
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    plt_size = 10
    fig, ax = plt.subplots(figsize=(plt_size,plt_size)) 
    plt.xlim(-1.25, 1.25)
    plt.ylim(-1.25, 1.25)

    header_path = "assets/Stratum2-Medium.otf"
    head_font = fm.FontProperties(fname=header_path)

    body_path = "assets/KievitOffc.ttf"
    body_font = fm.FontProperties(fname=body_path)

    # draw the unit circle
    fig = plt.gcf()
    ax = fig.gca()
    circle1 = plt.Circle((0, 0), 1.0, color='0.25', fill=False)
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
    ax.axhline(0, color="black")
    ax.axvline(0, color="black")

    # print emotion labels
    ax.text(0.98, 0.35, 'Happy', size=int(plt_size*2.5), fontproperties=body_font)
    ax.text(0.5, 0.9, 'Excited', size=int(plt_size*2.5), fontproperties=body_font)
    ax.text(-1.16, 0.35, 'Afraid', size=int(plt_size*2.5), fontproperties=body_font)
    ax.text(-0.7, 0.9, 'Angry', size=int(plt_size*2.5), fontproperties=body_font)
    ax.text(-1.13, -0.25, 'Sad', size=int(plt_size*2.5), fontproperties=body_font)
    ax.text(-0.9, -0.9, 'Depressed', size=int(plt_size*2.5), fontproperties=body_font)
    ax.text(0.98, -0.25, 'Content', size=int(plt_size*2.5), fontproperties=body_font)
    ax.text(0.7, -0.9, 'Calm', size=int(plt_size*2.5), fontproperties=body_font)

    plt.savefig(fname)
    plt.clf()

def main():
    args = parseargs()
    df = pd.read_csv(args.path)
    circumplex_model(df, args.title, args.fname)

def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a circumplex for valence/arousal information")
    parser.add_argument("-p", dest="path", type=str, help='A path to the csv containint valence/arousal values to be plotted')
    parser.add_argument("-o", dest="fname", type=str, help='The file name of the output png')
    parser.add_argument("-t", dest="title", type=str, help='The title of the graph')
    return parser.parse_args()

if __name__ == "__main__":
    main()