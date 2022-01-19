import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from typing import Tuple
from transformers.models.distilbert.tokenization_distilbert import DistilBertTokenizer
from transformers.utils.dummy_tf_objects import TFDistilBertForSequenceClassification
from transformers.models.distilbert.configuration_distilbert import DistilBertConfig
from feature_engineering.song_loader import get_song_df


distil_bert = 'bhadresh-savani/distilbert-base-uncased-emotion'

def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature extraction for social media sentiment using BERT."
    )
    parser.add_argument('-i', '--input_dir', dest='input', type=str,
        help = "Path to the directory storing the JSON files for social media data.")
    parser.add_argument('--source', required=True, type=str, dest='sm_type', 
        help = "Which type of social media input is being delivered.\n\
            Valid options are [Twitter, Youtube, Reddit, Lyrics].")
    parser.add_argument('--dataset', type=str, dest='dataset', required=True,
        help = "Name of the dataset which the comments represent")
    return parser.parse_args()

def tokenize(comment: str, tokenizer) -> Tuple[int, int, int]:
    return tokenizer.encode_plus(comment, add_special_tokens=True,
        return_attention_mask=True, return_token_type_ids=True)

def extract_ids(embedding: dict):
    return embedding['input_ids'], embedding['attention_mask'], embedding['token_type_ids']

def generate_embeddings(df: pd.DataFrame, tokenizer):
    inputs = map(lambda x: tokenize(x, tokenizer), tqdm(df['body']))
    df['input_ids'], df['input_masks'], df['input_segments'] = map(extract_ids, inputs)

def main():
    args = parseargs()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    song_df = get_song_df(args.input)
    tokenizer = DistilBertTokenizer.from_pretrained(distil_bert,
        do_lower_case=True, add_special_tokens=True)

    song_df = generate_embeddings(song_df, tokenizer)

    config = DistilBertConfig(num_labels=6)
    config.output_hidden_states = False
    transformer_model = TFDistilBertForSequenceClassification.from_pretrained(distil_bert, config = config)[0]
    
    input_ids = tf.keras.layers.Input(shape=(128,), name='input_token', dtype='int32')
    input_masks_ids = tf.keras.layers.Input(shape=(128,), name='masked_token', dtype='int32')
    X = transformer_model(input_ids, input_masks_ids)
    model = tf.keras.Model(inputs=[input_ids, input_masks_ids], outputs = X)

    embeddings = song_df[['input_ids', 'input_masks', 'input_segments']].apply(lambda x: x.to_numpy, axis=1)

    print(embeddings)
    
