import argparse
import functools
import os
from typing import List, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def default_load(
    path: str, ds_name: str, label_type: str = "Dimensional (Valence, Arousal)", dropcols: Union[None, List] = None
) -> pd.DataFrame:
    df = pd.DataFrame(pd.read_csv(path))
    if dropcols:
        df.drop(dropcols, axis=1, inplace=True)
    df["Dataset"] = ds_name
    df["label_type"] = label_type
    return df


def process_deezer_subset(subset: str, path: str, label_type: str) -> pd.DataFrame:
    path = path.split(".")[0]
    df = pd.DataFrame(pd.read_csv(f"{path}_{subset}.csv"))
    df["Dataset"] = "deezer"
    df["label_type"] = label_type
    df["subset"] = subset
    return df


def deezer_load(
    path: str, ds_name: str, label_type: str = "Dimensional (Valence, Arousal)", dropcols: Union[None, List] = None
) -> pd.DataFrame:
    deezer_apply = functools.partial(process_deezer_subset, path=path.split("_")[0], label_type=label_type)
    df = pd.concat(list(map(deezer_apply, ["test", "train", "validation"])))
    return df


loaders = {
    "amg1608": functools.partial(default_load, dropcols=["Genre"]),
    "deam": default_load,
    "pmemo": functools.partial(default_load, dropcols=["Genre"]),
    "deezer": deezer_load,
}


def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate whisker plots and circumplexes in one pass.")
    parser.add_argument("-i", dest="input", required=True, help="Input path for CSV")
    parser.add_argument("--stage_name", dest="stage_name", required=True)
    return parser.parse_args()


def circumplex_model(df: pd.DataFrame, title, fname) -> None:
    # Attempt to emulate sns fonts
    # mpl.rcParams["font.family"] = "Arial"
    # sns.set(font_scale=1.25)
    sns.set_style("whitegrid", {"axes.grid": False})
    plt_size = 10
    fig, ax = plt.subplots(figsize=(plt_size, plt_size))
    plt.xlim(-0.12, 1.12)
    plt.ylim(-0.12, 1.12)

    # draw the unit circle
    fig = plt.gcf()
    ax = fig.gca()
    circle1 = plt.Circle((0.5, 0.5), 0.5, color="0.25", fill=False)
    # ax.add_artist(circle1)

    ax = sns.scatterplot(data=df, x="valence", y="arousal", color="orange", alpha=0.50)
    ax.add_patch(circle1)
    ax.set_aspect("equal")
    # ax.scatter(val, aro, color="orange", alpha=0.5, s=20)
    # ax.grid(True)

    ax.set_xlabel("Valence", size=plt_size * 2.5, labelpad=15)
    ax.set_ylabel("Arousal", size=plt_size * 2.5, labelpad=15)
    ax.set_title(title, size=plt_size * 3)

    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.axhline(0.5, linewidth=1.0, color="black")
    ax.axvline(0.5, linewidth=1.0, color="black")

    # print emotion labels
    ax.text(rescale(1.00), rescale(0.35), "Happy", size=int(plt_size * 2.0))
    ax.text(rescale(0.5), rescale(0.9), "Excited", size=int(plt_size * 2.0))
    ax.text(rescale(-1.23), rescale(0.35), "Afraid", size=int(plt_size * 2.0))
    ax.text(rescale(-0.78), rescale(0.9), "Angry", size=int(plt_size * 2.0))
    ax.text(rescale(-1.20), rescale(-0.25), "Sad", size=int(plt_size * 2.0))
    ax.text(rescale(-1.09), rescale(-0.9), "Depressed", size=int(plt_size * 2.0))
    ax.text(rescale(1.0), rescale(-0.25), "Content", size=int(plt_size * 2.0))
    ax.text(rescale(0.66), rescale(-0.9), "Calm", size=int(plt_size * 2.0))

    # ax.set_aspect('equal')
    plt.savefig(fname)
    plt.show()
    plt.clf()


def rescale(i):
    return (0) + ((1 - (0)) / (1 - (-1))) * (i - (-1))


def df_norm(df: pd.DataFrame) -> pd.DataFrame:
    mms = MinMaxScaler()
    df[["valence", "arousal"]] = mms.fit_transform(df[["valence", "arousal"]])
    return df


def new_adjust_box_widths(axes, fac=0.9):
    """
    Adjust the widths of a seaborn-generated boxplot or boxenplot.

    Notes
    -----
    - thanks https://github.com/mwaskom/seaborn/issues/1076
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import PathPatch

    if isinstance(axes, list) is False:
        axes = [axes]

    # iterating through Axes instances
    for ax in axes:
        # iterating through axes artists:
        for c in ax.get_children():
            # searching for PathPatches
            if isinstance(c, PathPatch) or isinstance(c, PatchCollection):
                if isinstance(c, PathPatch):
                    p = c.get_path()
                else:
                    p = c.get_paths()[-1]

                # getting current width of box:
                #                 p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    try:
                        if np.all(l.get_xdata() == [xmin, xmax]):
                            l.set_xdata([xmin_new, xmax_new])
                    except:
                        # /tmp/ipykernel_138835/916607433.py:32: DeprecationWarning: elementwise comparison failed;
                        # this will raise an error in the future.
                        # if np.all(l.get_xdata() == [xmin, xmax]):
                        pass
    pass


def main() -> None:
    args = parseargs()

    sns.color_palette("rocket", as_cmap=True)
    #  print( [args.input+x for x in  os.listdir(args.input)])
    ds_name = os.path.basename(args.input).split(".")[0].lower()
    df = loaders[ds_name.split("_")[0]](path=args.input, ds_name=ds_name)
    print(df)

    # if os.path.isdir(args.input):
    #     any(map(functools.partial(insert_songs, collection=songs), [args.input+x for x in  os.listdir(args.input)]))
    # else:
    #     insert_songs(args.input, collection=songs)
    dfs = pd.concat(
        list(
            map(
                df_norm,
                map(
                    lambda x: x.reset_index(drop=True),
                    [
                        loaders[x](path=os.path.dirname(args.input) + f"{x.upper()}.csv", ds_name=x)
                        for x in loaders.keys()
                    ],
                ),
            )
        ),
        axis=0,
        ignore_index=True,
    )
    print(dfs)

    dfs = dfs.rename(columns={"valence": "Valence", "arousal": "Arousal"})
    mfs = pd.melt(dfs, id_vars=["Dataset"], value_vars=["Valence", "Arousal"])
    print(mfs)
    sns.set(font_scale=1.25)
    plt_size = 10
    fig, ax = plt.subplots(figsize=(plt_size, plt_size))
    ax = sns.boxplot(data=mfs, x="Dataset", y="value", hue="variable", palette="pastel")
    ax.set_title("Dataset Distributions", size=25)
    ax.set_ylabel("")
    ax.set_xlabel("")
    new_adjust_box_widths(ax)
    plt.xticks([0, 1, 2, 3], ["AMG1608", "DEAM", "PMEmo", "Deezer"])
    ax.tick_params(axis="x", which="major", labelsize=18)
    sns.move_legend(ax, "upper right")
    plt.show()
    # mdf = pd.melt(dfs, id_vars)

    circumplex_model(df_norm(df)[["valence", "arousal"]], f"{args.stage_name}", f"circumplex_{ds_name}.png")


if __name__ == "__main__":
    main()
