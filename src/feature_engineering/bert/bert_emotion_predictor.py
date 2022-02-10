import argparse
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
import configparser
import neptune.new as neptune 

from typing import Tuple
from transformers import DistilBertTokenizer

from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from tf.keras.callbacks import ModelCheckpoint

from feature_engineering.song_loader import get_song_df
from model_factory import create_model

# - load data from json 
# - remove url's, html tags (maybe use that regex from wordbag?)
# - tokenize
# - convert attn. mask, input IDs, and label into tensor dataset
# - build model architecture
# -   -   adam optimizer with 5e-5 learn rate 
# -   -   distilBERT with regression head on top (yes, i know it's called classification)
# -   -    -    since it's a multi-target regression task, uses cross-entropy as loss function

distil_bert = 'distilbert-base-uncased'
MAX_SEQ_LEN = 128


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
    parser.add_argument('-c', '--config', type=str, dest='config', required=True,
        help="Credentials file for Neptune.AI")
    parser.add_argument('m', '--model', type=str, dest='model', required=True,
        help="Path to saved model state, if model doesn't exist at path, creates a new checkpoint.")
    return parser.parse_args()


def tokenize(comments: pd.Series, tokenizer) -> transformers.BatchEncoding:
    return tokenizer(list(comments), add_special_tokens=True, return_attention_mask=True,
                    return_token_type_ids=False, max_length=MAX_SEQ_LEN, padding='max_length',
                    truncation=True, return_tensors='tf')


def generate_embeddings(df: pd.DataFrame) -> tf.data.Dataset:
    # Initialize tokenizer - set to automatically lower-case
    tokenizer = DistilBertTokenizer.from_pretrained(distil_bert,
        do_lower_case=True, add_special_tokens=True, max_length=MAX_SEQ_LEN, padding='max_length', truncate=True, padding_side='right')
    encodings = tokenize(df['body'], tokenizer)
    return tf.data.Dataset.from_tensor_slices(({
        'input_token': encodings['input_ids'],
        'masked_token': encodings['attention_mask'],
    }, tf.constant((df[['valence', 'arousal']].values).astype('float32')))) 


def get_num_gpus() -> int:
    return len(tf.config.list_physical_devices('GPU'))

def _process_api_key(f_key: str) -> configparser.ConfigParser:
    api_key = configparser.ConfigParser()
    api_key.read(f_key)
    return api_key

def tf_config() -> Tuple[tf.distribute.Strategy, tf.data.Options]: 
    print(f"Num GPUs Available: {get_num_gpus()}")
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    tf.debugging.set_log_device_placement(True)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    return strategy, options

def init_neptune(cfg: str):
    creds = _process_api_key(cfg)
    runtime = neptune.init(project=creds['CLIENT_INFO']['project_id'],
                        api_token=creds['CLIENT_INFO']['api_token'])
    return NeptuneCallback(run=runtime, base_namespace='metrics')

def load_model(path: str) -> tf.keras.Model:
    try:
        return tf.keras.models.load_model(path)
    except IOError:
        return create_model()

def main():
    args = parseargs()
    distribution_strategy, ds_options  = tf_config()
    # load neptune callback for keras
    callbacks = [init_neptune(args.config), 
                 ModelCheckpoint(args.model, monitor='loss',
                                 verbose=1, save_best_only=True, mode='min')]

    # Load our data from JSONs
    song_df = get_song_df(args.input)

    # Clean strings - remove urls and html tags
    rx = re.compile(r'(?:<.*?>)|(?:http\S+)')
    song_df['body'] = song_df['body'].apply(lambda x: rx.sub('', x))

    # Create tf.Dataset with input ids, attention mask, and [valence, arousal] target labels
    # TODO - need a train-test split
    song_data_encodings = generate_embeddings(song_df)

    # Batch our dataset according to available resources
    # IMPORTANT: MUST drop remainder in order to prevent segfault when training with multiple GPUs + cuDNN kernel function
    song_data_encodings = song_data_encodings.batch(64 * get_num_gpus(), drop_remainder=True)
    song_data_encodings = song_data_encodings.with_options(ds_options)

    with distribution_strategy.scope():
        model = load_model(args.model)
        print(model.summary())
        model.fit(song_data_encodings, verbose=1, epochs=50, callbacks=callbacks)

    model.save('reddit_amg_model')

    # TODO - predictions
    # TODO - ensure one song per inference
    
    
