import transformers
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from transformers import DistilBertTokenizerFast

distil_bert = 'distilbert-base-uncased'
RAND_SEED = 128

def _tokenize(comments: pd.Series, tokenizer, seq_len: int) -> transformers.BatchEncoding:
    return tokenizer(list(comments),
                     add_special_tokens=True,
                     return_attention_mask=True,
                     return_token_type_ids=False,
                     max_length=seq_len,
                     padding='max_length',
                     truncation=True,
                     return_tensors='tf')


def generate_embeddings(df: pd.DataFrame, seq_len: int) -> dict:
    tokenizer = DistilBertTokenizerFast.from_pretrained(distil_bert,
                                                        do_lower_case=True,
                                                        add_special_tokens=True,
                                                        max_length=seq_len,
                                                        padding='max_length',
                                                        truncate=True,
                                                        padding_side='right')

    encodings = _tokenize(df['body'], tokenizer, seq_len)
    return {'input_token': encodings['input_ids'],
            'masked_token': encodings['attention_mask']}


class DiscourseDataSet:
    def __init__(self, df: pd.DataFrame, t_prop: float):
        self.df = self._clean_str(df)
        # TODO - introduce validation subset
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self._split_data(self.df,
                                                                                                        test_size=t_prop)

    # NOTE - ONLY cleans comment bodies. Adapt to post titles if needed.
    def _clean_str(self, df: pd.DataFrame):
        rx = re.compile(r'(?:<.*?>)|(?:http\S+)')
        df['body'] = df['body'].apply(lambda x: rx.sub('', x))
        return df


    def _split_data(self, df: pd.DataFrame, test_size):
        np.random.seed(RAND_SEED)
        ids = df['song_name'].unique()
        holdout_indices = np.random.choice(ids, size=int(len(ids) * test_size), replace=False)
        train_subset = df.loc[~df['song_name'].isin(holdout_indices)]
        test_indices = np.random.choice(holdout_indices, size=int(len(holdout_indices) * 0.5), replace=False)
        test_subset = df.loc[df['song_name'].isin(test_indices)]
        validation_subset = df.loc[~df['song_name'].isin(test_indices)]
        print(test_subset.shape)
        print(train_subset.shape)
        print(validation_subset.shape)
        return self._features(train_subset),\
            self._features(validation_subset),\
            self._features(test_subset),\
            self._convert_labels(train_subset),\
            self._convert_labels(validation_subset),\
            self._convert_labels(test_subset)


    def _features(self, a: pd.DataFrame) -> pd.DataFrame:
        return a.drop(['valence', 'arousal'], axis=1)


    def _convert_labels(self, a: pd.DataFrame):
        a = a[['valence', 'arousal']]
        scaler = MinMaxScaler(feature_range=(0,1))
        a = scaler.fit_transform(a)
        return np.asarray(a).astype('float32')
