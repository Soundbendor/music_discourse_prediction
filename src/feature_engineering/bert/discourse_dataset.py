import transformers
import tensorflow as tf
import pandas as pd

from tensorflow.data import Dataset as TFDataset
from typing import Tuple
from transformers import DistilBertConfig
from transformers import DistilBertTokenizer

distil_bert = 'distilbert-base-uncased'


class DiscourseDataSet:

    def __init__(self, df: pd.DataFrame, num_labels: int, seq_len: int, test_prop: float, batch_size: int, options: tf.data.Options) -> None:
        self.num_labels = num_labels
        self.seq_len = seq_len
        self.test_prop = test_prop
        self.batch_size = batch_size
        self.options = options
        self.train, self.test, self.validate = self._to_datasets(df)
        

    def _to_datasets(self, df: pd.DataFrame) -> Tuple[TFDataset, TFDataset, TFDataset]:
        ds = self._generate_embeddings(df)
        ds_train = ds.skip(int(2*(df.shape[1] * self.test_prop)))
        # we derive validation and test from holdout - this is a temp variable
        ds_holdout = ds.take(int(2*(df.shape[1] * self.test_prop)))
        ds_test, ds_validate = ds_holdout.shard(2, 0), ds_holdout.shard(2, 1)
        # IMPORTANT: MUST drop remainder in order to prevent segfault when training with multiple GPUs + cuDNN kernel function
        return tuple(map(lambda x: x.batch(self.batch_size, drop_remainder=True).with_options(self.options), [ds_train, ds_test, ds_validate]))


    def _tokenize(self, comments: pd.Series, tokenizer) -> transformers.BatchEncoding:
        return tokenizer(list(comments),
            add_special_tokens = True,
            return_attention_mask = True,
            return_token_type_ids = False,
            max_length = self.seq_len,
            padding = 'max_length',
            truncation = True,
            return_tensors = 'tf')


    def _generate_embeddings(self, df: pd.DataFrame) -> tf.data.Dataset:
        tokenizer = DistilBertTokenizer.from_pretrained(distil_bert,
            do_lower_case=True,
            add_special_tokens=True,
            max_length=self.seq_len,
            padding='max_length',
            truncate=True,
            padding_side='right')

        encodings = self._tokenize(df['body'], tokenizer)
        return tf.data.Dataset.from_tensor_slices(({
            'input_token': encodings['input_ids'],
            'masked_token': encodings['attention_mask']},
            tf.constant((df[['valence', 'arousal']].values).astype('float32')))) 