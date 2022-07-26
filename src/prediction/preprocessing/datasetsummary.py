class DatasetSummary:
    def __init__(self, meta_df, n_features) -> None:
        self.meta_df = meta_df
        self.n_features = n_features

    def get_songset_name(self) -> str:
        print(self.meta_df)
        return self.meta_df['dataset'].iloc[[0]]

    def get_n_examples(self) -> int:
        return self.meta_df.shape[0]

    def get_n_features(self) -> int:
        return self.n_features

    def get_n_comments(self) -> int:
        return int(sum(self.meta_df['submission.n_comments']))

    def get_n_words(self) -> int:
        return int(sum(self.meta_df['n_words']))
