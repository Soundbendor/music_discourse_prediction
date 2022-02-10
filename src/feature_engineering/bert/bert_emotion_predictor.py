import argparse
import re

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from feature_engineering.song_loader import get_song_df
from model_factory import create_model, generate_embeddings
from tf_configurator import get_num_gpus, init_neptune, tf_config

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
    
    