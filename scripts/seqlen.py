from typing import List, Union

import seaborn as sns

from database.driver import Driver

SOURCES = ["Reddit", "Youtube", "Twitter"]
DATASET = ["deam_new" "amg1608", "pmemo"]
# Averages for all individual sources and datasets


# Cumulative histograms
def make_hist(src: Union[List[str], str]) -> None:
    df = db_con.get_discourse(ds_name="amg1608", source_type=src)
    print(df["body"].str.len().describe())
    hist = sns.histplot(data=df["body"].str.len().clip(0, 1024), kde=True)
    hist.set(xlabel="Comment Length", ylabel="Songs")
    fig = hist.get_figure()
    fig.savefig(f"{src}_dist.png")


sns.color_palette("rocket", as_cmap=True)
db_con = Driver("mdp")
for source in SOURCES:
    make_hist(source)

make_hist(SOURCES)
