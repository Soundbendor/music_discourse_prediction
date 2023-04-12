from typing import List, Union

import matplotlib.pyplot as plt
import seaborn as sns

from database.driver import Driver

SOURCES = ["Reddit", "Youtube", "Twitter"]
DATASET = ["deam_new", "amg1608", "pmemo"]
# Averages for all individual sources and datasets


# Cumulative histograms
def make_hist(src: Union[List[str], str]) -> None:
    df = db_con.get_discourse(ds_name=DATASET, source_type=src)
    print(df["body"].str.len().describe())
    hist = sns.histplot(data=df["body"].str.len().clip(0, 1024), kde=True)
    return hist


sns.color_palette("rocket", as_cmap=True)
db_con = Driver("mdp")
hist = None
for source in SOURCES:
    hist = make_hist(source)

# Bad stupid code design
hist.set(yscale="log")
hist.set(xlabel="Comment Length", ylabel="Songs")
plt.legend(labels=SOURCES)
fig = hist.get_figure()
fig.savefig(f"all_dist.png")


# make_hist(SOURCES)
