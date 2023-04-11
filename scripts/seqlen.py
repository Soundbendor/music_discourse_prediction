import seaborn as sns

from database.driver import Driver

DATASET = ["deezer", "deam_new" "amg1608", "pmemo"]
# Averages for all individual sources and datasets

# Cumulative histograms

# Total final histogram
db_con = Driver("mdp")
for source in ["Reddit", "Youtube", "Twitter"]:
    df = db_con.get_discourse(ds_name="amg1608", source_type=source)
    print(df["body"].str.len().describe())
    hist = sns.histplot(data=df["body"].str.len().clip(0, 1024), kde=True, color="rocket")
    hist.set(xlabel="Comment Length", ylabel="Songs")
    fig = hist.get_figure()
    fig.savefig(f"{source}_dist.png")
