import argparse
import re
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd

from tensorflow.keras.callbacks import ModelCheckpoint

from feature_engineering.song_loader import get_song_df
from .model_assembler import create_direct_model
from .discourse_dataset import DiscourseDataSet
from .tf_configurator import get_num_gpus, init_neptune, tf_config


def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature extraction for social media sentiment using BERT."
    )
    parser.add_argument('-i', '--input_dir', dest='input', type=str,
                        help="Path to the directory storing the JSON files for social media data.")
    parser.add_argument('--source', required=True, type=str, dest='sm_type',
                        help="Which type of social media input is being delivered.\n\
            Valid options are [Twitter, Youtube, Reddit, Lyrics].")
    parser.add_argument('--dataset', type=str, dest='dataset', required=True,
                        help="Name of the dataset which the comments represent")
    parser.add_argument('-c', '--config', type=str, dest='config', required=True,
                        help="Credentials file for Neptune.AI")
    parser.add_argument('-m', '--model', type=str, dest='model', required=True,
                        help="Path to saved model state, if model doesn't exist at path, creates a new checkpoint.")
    parser.add_argument('--num_epoch', type=int, default=50, dest='num_epoch',
                        help="Number of epochs to train the model with")
    return parser.parse_args()


def load_weights(model: tf.keras.Model, path: str):
    try:
        model.load_weights(path)
    except Exception:
        print("Model checkpoint invalid. Opening new model.")


def main():
    args = parseargs()
    distribution_strategy, ds_options = tf_config()
    # load neptune callback for keras
    callbacks = [init_neptune(args.config),
                 ModelCheckpoint(args.model, monitor='loss',
                                 save_weights_only=True, verbose=1,
                                 save_best_only=True, mode='min')]

    # Load our data from JSONs and randomize the dataset
    # We shuffle here because tensorflow does not currently support dataset shuffling
    song_df = get_song_df(args.input).sample(frac=1)

    # Clean strings - remove urls and html tags
    rx = re.compile(r'(?:<.*?>)|(?:http\S+)')
    song_df['body'] = song_df['body'].apply(lambda x: rx.sub('', x))

    # TODO - find optimal token length
    ds = DiscourseDataSet(song_df,
                          num_labels=2,
                          seq_len=128,
                          test_prop=0.15,
                          batch_size=(64 * get_num_gpus()),
                          options=ds_options)

    with distribution_strategy.scope():
        model = create_direct_model()
        load_weights(model, args.model)
        print(model.summary())

        model.fit(ds.train, verbose=1, callbacks=callbacks,
                  epochs=args.num_epoch)
        model.save_weights('r_amg_model_finished')

        print("\n\nValidating...")
        model.evaluate(ds.validate, verbose=1, callbacks=callbacks)

        print("\n\nTesting...")
        # TODO
        y_pred = model.predict(ds.test, verbose=1, callbacks=callbacks)
        corr = tfp.stats.correlation(y_pred, ds.test)
        print(corr)
        pd.Dataframe(y_pred).to_csv("results.csv")

