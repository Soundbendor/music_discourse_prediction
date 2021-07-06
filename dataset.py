import pandas as pd
import numpy as np

from experimentfactory import ExperimentFactory
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, config: dict, fname: str) -> None:
        self.config = config
        self.fname = fname
        self.df = self._process_df()
        self.X, self.y = self.split_x_y(self.df)
        

    def _get_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self.fname, index_col=False)
    
    def _drop_at_threshold(self, df) -> pd.DataFrame:
        df.replace(0, np.nan, inplace=True)
        df.dropna(how='any', inplace=True, thresh= self.config['threshold'])
        df.fillna(0, inplace=True)
        return df

    # We separate this step from other preprocessing tasks because in the future,
    #  we may want to encode categorical data before dropping non-numeric metadata. 
    def _drop_nonnumeric(self, df) -> pd.DataFrame:
        return df.select_dtypes('number').copy()

    def _drop_meta(self, df) -> pd.DataFrame:
        return df.drop(self.config['meta_cols'], axis=1)

    def _process_df(self) -> pd.DataFrame:
        return self._drop_nonnumeric(
            self._drop_meta(
                self._drop_at_threshold(
                    self._get_dataframe()
                )
            )
        )

    def split_x_y(self, df):
        val_key = self.config['valence_key']
        aro_key = self.config['arousal_key']
        X, y = df.drop(['existing_valence', 'existing_arousal'], axis=1), df[[val_key, aro_key]]
        return X, y

