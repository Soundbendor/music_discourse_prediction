from .datasetsummary import DatasetSummary

import os, glob, fnmatch, re
from typing import Tuple
import pandas as pd
import numpy as np


class Dataset:

    class_names = {
        'happy': 0,
        'upset': 1,
        'depressed': 2,
        'calm': 3
    }

    label_key = 'class'

    def __init__(self, config: dict, fname: str) -> None:
        self.config = config
        self.fname = fname
        self.val_key = config['valence_key']
        self.aro_key = config['arousal_key']
        self.df, self.meta_df = self._process_df()
        print(self.df)

        self.summary = DatasetSummary(self.meta_df, self.df.shape[1])

    def _get_dataframe(self) -> pd.DataFrame:
        return self._label_data(pd.read_csv(self.fname, index_col=False))

    def _drop_at_threshold(self, df) -> pd.DataFrame:
        df.replace(0, np.nan, inplace=True)
        df.dropna(how='any', inplace=True, thresh=self.config['threshold'])
        df.fillna(0, inplace=True)
        print(df)
        return df

    # We separate this step from other preprocessing tasks because in the future,
    # we may want to encode categorical data before dropping non-numeric metadata.
    def _get_nonnumeric_cols(self, df) -> list:
        return list(df.select_dtypes(exclude='number').columns)

    def _get_meta_cols(self) -> list:
        return self.config['meta_cols']

    def _get_drop_cols(self, df) -> list:
        return np.append(self._get_meta_cols(), self._get_nonnumeric_cols(df))

    def _label_data(self, df) -> pd.DataFrame:
        
        df.loc[(df[self.val_key] >= 0.5) & (df[self.aro_key] >= 0.5),
               self.label_key] = self.class_names['happy']
        df.loc[(df[self.val_key] >= 0.5) & (df[self.aro_key] < 0.5),
               self.label_key] = self.class_names['upset']
        df.loc[(df[self.val_key] < 0.5) & (df[self.aro_key] < 0.5),
               self.label_key] = self.class_names['depressed']
        df.loc[(df[self.val_key] < 0.5) & (df[self.aro_key] >= 0.5),
               self.label_key] = self.class_names['calm']

        return df

    def _process_df(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = self._drop_at_threshold(self._get_dataframe())
        drop_cols = self._get_drop_cols(df)
        meta_cols = df[drop_cols]
        data = df.drop(drop_cols, axis=1)

        return data, meta_cols

    def split_x_y(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X, y = self.df.drop([self.val_key, self.aro_key, self.label_key], axis=1), self.df[[
            self.val_key, self.aro_key, self.label_key]]
        return X, y

    def get_data(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        X, y = self.split_x_y()
        return X.to_numpy(), y[key].to_numpy()
