import argparse
import re
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint

from feature_engineering.song_loader import get_song_df
from .model_assembler import create_direct_model, create_model
from .discourse_dataset import DiscourseDataSet
from .tf_configurator import get_num_gpus, init_neptune, tf_config


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
    parser.add_argument('-m', '--model', type=str, dest='model', required=True,
        help="Path to saved model state, if model doesn't exist at path, creates a new checkpoint.")
    parser.add_argument('--num_epoch', type=int, default=1, dest='num_epoch',
        help="Epoch to start on, when loading from a checkpoint. Defaults to 1.")
    return parser.parse_args()

def load_model(path: str) -> tf.keras.Model:
    try:
        return tf.keras.models.load_model(path)
    except IOError:
        print("Model checkpoint invalid. Opening new model.")
        return create_direct_model()


def main():
    args = parseargs()
    distribution_strategy, ds_options  = tf_config()
    # load neptune callback for keras
    callbacks = [init_neptune(args.config), 
                 ModelCheckpoint(args.model, monitor='loss',
                                 verbose=1, save_best_only=True, mode='min')]

    # Load our data from JSONs and randomize the dataset
    # We shuffle here because tensorflow does not currently support dataset shuffling
    song_df = get_song_df(args.input).sample(frac=1)

    # Clean strings - remove urls and html tags
    rx = re.compile(r'(?:<.*?>)|(?:http\S+)')
    song_df['body'] = song_df['body'].apply(lambda x: rx.sub('', x))

    ds = DiscourseDataSet(song_df,
        num_labels=2,
        seq_len=128,
        test_prop=0.15,
        batch_size=(64 * get_num_gpus()),
        options=ds_options)

    with distribution_strategy.scope():
        model = load_model(args.model)
        print(model.summary())
        model.fit(ds.train, verbose=1, epochs=50, callbacks=callbacks, initial_epoch=args.num_epoch)

        model.save('reddit_amg_model')
        model.evaluate(ds.validate, verbose=1, callbacks=callbacks)

    
    