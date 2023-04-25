import re
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

RAND_SEED = 128


def _tokenize(comments: pd.Series, tokenizer, seq_len: int) -> BatchEncoding:
    return tokenizer(
        list(comments),
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        max_length=seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="tf",
    )


def generate_embeddings(df: pd.DataFrame, seq_len: int, model: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(
        model, add_special_tokens=True, max_length=seq_len, padding="max_length", truncate=True, padding_side="right"
    )

    encodings = _tokenize(df["body"], tokenizer, seq_len)
    return {"input_token": encodings["input_ids"], "masked_token": encodings["attention_mask"]}


class DiscourseDataSet:
    def __init__(self, df: pd.DataFrame, t_prop: float, score_threshold: int, length_threshold: int):
        df["body"].replace("", np.nan, inplace=True)
        self.df = self._clean_str(df.dropna(how="any", subset=["body"]), score_threshold, length_threshold)
        # TODO - introduce validation subset
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self._split_data(
            self.df, test_size=t_prop
        )

    # NOTE - ONLY cleans comment bodies. Adapt to post titles if needed.
    def _clean_str(self, df: pd.DataFrame, score_threshold: int, length_threshold: int):
        rx = re.compile(r"(?:<.*?>)|(?:http\S+)")
        df["body"] = df["body"].apply(lambda x: rx.sub("", x))
        # Drop duplicate comments
        df = df.drop_duplicates(subset="body")
        # Filter by score
        df = df.drop(df[df.score < score_threshold].index)
        # filter by length
        df = df.drop(df[df["body"].map(len) < length_threshold].index)
        print(f"Dataframe Length: {len(df)}")
        return df

    def _split_data(self, df: pd.DataFrame, test_size):
        np.random.seed(RAND_SEED)
        holdout_train_split = self._split_songs(df, test_size)
        test_validate_split = self._split_songs(holdout_train_split["Holdout"], 0.5)

        y_train, y_val, y_test = self._convert_labels(
            holdout_train_split["Include"], test_validate_split["Include"], test_validate_split["Holdout"]
        )

        return (
            self._features(holdout_train_split["Include"]),
            self._features(test_validate_split["Include"]),
            self._features(test_validate_split["Holdout"]),
            y_train,
            y_val,
            y_test,
        )

    def _split_songs(self, df: pd.DataFrame, test_size: float) -> dict:
        ids = df["song_name"].unique()
        holdout_indices = np.random.choice(ids, size=int(len(ids) * test_size), replace=False)
        holdout_df = df.loc[df["song_name"].isin(holdout_indices)]
        include_df = df.loc[~df["song_name"].isin(holdout_indices)]
        return {"Holdout": holdout_df, "Include": include_df}

    def _features(self, x: pd.DataFrame) -> pd.DataFrame:
        return x.drop(["valence", "arousal"], axis=1)  # type: ignore

    def _convert_labels(
        self, y_train: pd.DataFrame, y_val: pd.DataFrame, y_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        scaler = MinMaxScaler(feature_range=(0, 1))
        y_train = scaler.fit_transform(self._get_labels(y_train))
        y_val = scaler.transform(self._get_labels(y_val))
        y_test = scaler.transform(self._get_labels(y_test))
        return y_train, y_val, y_test

    def _get_labels(self, x: pd.DataFrame) -> pd.DataFrame:
        return x[["valence", "arousal"]]  # type: ignore

    def _to_float32(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x).astype("float32")  # type: ignore
