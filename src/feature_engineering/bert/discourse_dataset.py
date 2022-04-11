import transformers
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast

distil_bert = 'distilbert-base-uncased'


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
        self.X = df.drop(['valence', 'arousal'], axis=1)
        self.y = df[['valence', 'arousal']]
        self.X_train, self.X_test, self.y_train, self.y_test = self._split_data(self.X,
                                                                                self.y,
                                                                                test_size=t_prop)

    # NOTE - ONLY cleans comment bodies. Adapt to post titles if needed.
    def _clean_str(self, df: pd.DataFrame):
        rx = re.compile(r'(?:<.*?>)|(?:http\S+)')
        df['body'] = df['body'].apply(lambda x: rx.sub('', x))
        return df

    def _split_data(self, X, y, test_size):
        X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=test_size)
        return X_train, X_test, self._convert_labels(y_train), self._convert_labels(y_test)

    def _convert_labels(self, a):
        return np.asarray(a).astype('float32')
