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
        self.df = self._process_df()

       

    def _get_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self.fname, index_col=False)
    
    def _drop_at_threshold(self, df) -> pd.DataFrame:
        df.replace(0, np.nan, inplace=True)
        df.dropna(how='any', inplace=True, thresh= self.config['threshold'])
        df.fillna(0, inplace=True)
        return df

    # We separate this step from other preprocessing tasks because in the future,
    # we may want to encode categorical data before dropping non-numeric metadata. 
    def _drop_nonnumeric(self, df) -> pd.DataFrame:
        return df.select_dtypes('number').copy()

    def _drop_meta(self, df) -> pd.DataFrame:
        return df.drop(self.config['meta_cols'], axis=1)

    def _label_data(self, df) -> pd.DataFrame: 

        df.loc[(df[self.val_key] >= 0) & (df[self.aro_key] >= 0), self.label_key] = self.class_names['happy']
        df.loc[(df[self.val_key] >= 0) & (df[self.aro_key] < 0), self.label_key] = self.class_names['upset']
        df.loc[(df[self.val_key] < 0) & (df[self.aro_key] < 0), self.label_key] = self.class_names['depressed']
        df.loc[(df[self.val_key] < 0) & (df[self.aro_key] >= 0), self.label_key] = self.class_names['calm']

        return df

    def _process_df(self) -> pd.DataFrame:
        return self._label_data(
            self._drop_nonnumeric(
                self._drop_meta(
                    self._drop_at_threshold(
                        self._get_dataframe()
                    )
                )
            )
        )

    def split_x_y(self):
        X, y = self.df.drop([self.val_key, self.aro_key, self.label_key], axis=1), self.df[[self.val_key, self.aro_key, self.label_key]]
        return X, y

    def get_data(self, key: str ):
        X, y = self.split_x_y()
        return X.values, y[key].values


