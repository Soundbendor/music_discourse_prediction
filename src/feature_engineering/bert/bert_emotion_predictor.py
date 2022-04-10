import argparse
import os
import tensorflow as tf
import neptune.new as neptune

from dotenv import load_dotenv
from tensorflow.keras.callbacks import ModelCheckpoint
from scipy.stats import pearsonr
from feature_engineering.song_loader import get_song_df
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

from .discourse_dataset import DiscourseDataSet, generate_embeddings
from .model_assembler import create_model

SEQ_LEN = 128
BATCH_SIZE = 64


def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature extraction for social media sentiment using BERT.")
    parser.add_argument('-i', '--input_dir', dest='input', type=str,
                        help="Path to the directory storing the JSON files for social media data.")
    parser.add_argument('--source', required=True, type=str, dest='sm_type',
                        help="Which type of social media input is being delivered.\n\
            Valid options are [Twitter, Youtube, Reddit, Lyrics].")
    parser.add_argument('--dataset', type=str, dest='dataset', required=True,
                        help="Name of the dataset which the comments represent")
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


def get_num_gpus() -> int:
    return len(tf.config.list_physical_devices('GPU'))


def init_neptune():
    load_dotenv()
    runtime = neptune.init(project=os.getenv('NEPTUNE_PROJECT_ID'),
                           api_token=os.getenv('NEPTUNE_API_TOKEN'))
    return NeptuneCallback(run=runtime, base_namespace='metrics')


def main():
    args = parseargs()
    # load neptune callback for keras
    callbacks = [init_neptune(),
                 ModelCheckpoint(args.model, monitor='loss',
                                 save_weights_only=True, verbose=1,
                                 save_best_only=True, mode='min')]

    # Load our data from JSONs and randomize the dataset
    # We shuffle here because tensorflow does not currently support dataset shuffling
    song_df = get_song_df(args.input)

    ds = DiscourseDataSet(song_df, t_prop=0.15)

    with tf.distribute.MultiWorkerMirroredStrategy().scope():
        model = create_model()
        load_weights(model, args.model)
        print(model.summary())

        print(ds.X_train)
        print(ds.X_train.shape)
        print(ds.y_train)
        print(ds.y_train.shape)

        model.fit(x=generate_embeddings(ds.X_train, SEQ_LEN),
                  y=ds.y_train,
                  verbose=1,
                  batch_size=(BATCH_SIZE * get_num_gpus()),
                  callbacks=callbacks,
                  epochs=args.num_epoch)

        print("\n\nTesting...")
        y_pred = model.predict(x=generate_embeddings(ds.X_test, SEQ_LEN),
                               batch_size=(BATCH_SIZE * get_num_gpus()),
                               verbose=1,
                               callbacks=callbacks)

        print(y_pred)
        print(y_pred.shape)

        print(ds.X_test)
        print(ds.y_test)
        print(ds.y_test.shape)
        
        valence_corr = pearsonr(ds.y_test[:, 0], y_pred[:, 0])
        arr_corr = pearsonr(ds.y_test[:, 1], y_pred[:, 1])
        print(f"Pearson's Correlation - Valence: {valence_corr}")
        print(f"Pearson's Correlation - Arousal: {arr_corr}")
