import argparse
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import neptune.new as neptune
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from tensorflow.keras.callbacks import ModelCheckpoint
from scipy.stats import pearsonr
from feature_engineering.song_loader import get_song_df
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

from .discourse_dataset import DiscourseDataSet, generate_embeddings
from prediction.visualization.visualizations import circumplex_model
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
    parser.add_argument('--model_name', type=str,
                        default='distilbert-base-cased', dest='model_name')
    parser.add_argument('--intersection_type', type=str,
                        default='NA', dest='intersection_type')
    return parser.parse_args()


def load_weights(model: tf.keras.Model, path: str):
    try:
        model.load_weights(path)
    except Exception:
        print("Model checkpoint invalid. Opening new model.")


def get_num_gpus() -> int:
    return len(tf.config.list_physical_devices('GPU'))


def get_songs(args: argparse.Namespace):
    if args.intersection_type == 'intersect':
        reddit = get_song_df(
            f"{args.input}/reddit").dropna(how='any', subset=['body'])
        twitter = get_song_df(
            f"{args.input}/twitter").dropna(how='any', subset=['body'])
        if args.dataset == 'deezer':
            df = pd.concat([reddit, twitter])
            return df[(df['query_index'].isin(twitter['query_index'])) &
                      (df['query_index'].isin(reddit['query_index']))]
        youtube = get_song_df(
            f"{args.input}/youtube").dropna(how='any', subset=['body'])
        df = pd.concat([reddit, twitter, youtube])
        return df[(df['query_index'].isin(twitter['query_index'])) &
                  (df['query_index'].isin(reddit['query_index'])) &
                  (df['query_index'].isin(youtube['query_index']))]
    return get_song_df(args.input)


def main():
    args = parseargs()
    load_dotenv()
    # load neptune callback for keras
    neptune_runtime = neptune.init(project=os.getenv('NEPTUNE_PROJECT_ID'),
                                   api_token=os.getenv('NEPTUNE_API_TOKEN'))
    callbacks = [NeptuneCallback(run=neptune_runtime, base_namespace='metrics'),
                 ModelCheckpoint(args.model, monitor='loss',
                                 save_weights_only=True, verbose=1,
                                 save_best_only=True, mode='min')]

    # Load our data from JSONs and randomize the dataset
    # We shuffle here because tensorflow does not currently support dataset shuffling
    song_df = get_songs(args)

    ds = DiscourseDataSet(song_df, t_prop=0.15)

    with tf.distribute.MultiWorkerMirroredStrategy().scope():
        model = create_model(args.model_name)
        load_weights(model, args.model)
        print(model.summary())

        model.fit(x=generate_embeddings(ds.X_train, SEQ_LEN, args.model_name),
                  y=ds.y_train,
                  verbose=1,
                  batch_size=(BATCH_SIZE * get_num_gpus()),
                  validation_data=(generate_embeddings(
                      ds.X_val, SEQ_LEN, args.model_name), ds.y_val),
                  callbacks=callbacks,
                  epochs=args.num_epoch)

        print("\n\nTesting...")
        y_pred = model.predict(x=generate_embeddings(ds.X_test, SEQ_LEN, args.model_name),
                               batch_size=(BATCH_SIZE * get_num_gpus()),
                               verbose=1,
                               callbacks=callbacks)

        valence_corr = pearsonr(ds.y_test[:, 0], y_pred[:, 0])
        arr_corr = pearsonr(ds.y_test[:, 1], y_pred[:, 1])

        print(
            f"Pearson's Correlation (comment level) - Valence: {valence_corr}")
        print(f"Pearson's Correlation (comment level) - Arousal: {arr_corr}")

        aggregate_predictions(ds.X_test, ds.y_test, y_pred, neptune_runtime,
                              f"{args.sm_type}_{args.dataset}")


def aggregate_predictions(X: pd.DataFrame, y: np.ndarray, pred: np.ndarray, run: neptune.Run, fname: str):
    X['valence'] = y[:, 0]
    X['arousal'] = y[:, 1]
    X['val_pred'] = pred[:, 0]
    X['aro_pred'] = pred[:, 1]
    print(X[['valence', 'arousal', 'val_pred', 'aro_pred']].describe())
    results = X.groupby(['song_name'])[
        ['valence', 'arousal', 'val_pred', 'aro_pred']].mean()

    results.to_csv(f"{fname}_results.csv")

    valence_corr = pearsonr(results['valence'], results['val_pred'])
    arr_corr = pearsonr(results['arousal'], results['aro_pred'])

    run['valence_p'] = valence_corr
    run['arousal_p'] = arr_corr

    print(f"Pearson's Correlation (song level) - Valence: {valence_corr}")
    print(f"Pearson's Correlation (song level) - Arousal: {arr_corr}")

    scatterplot(results, 'valence', 'val_pred',
                'valence_scatter', 'Valence', run)
    scatterplot(results, 'arousal', 'aro_pred',
                'arousal_scatter', 'Arousal', run)

    circumplex_model(results, f"{fname} - Predicted",
                     fname=f"{fname}_predicted.png", val_key='val_pred', aro_key='aro_pred')
    circumplex_model(results, f"{fname} - Actual",
                     fname=f"{fname}_actual.png", val_key='valence', aro_key='arousal')


def scatterplot(df: pd.DataFrame, x_key: str, y_key: str, fname: str, title: str, run: neptune.Run) -> None:
    fig = plt.figure()
    plt.scatter(x=df[x_key], y=df[y_key], alpha=0.5, s=10)
    m, b = np.polyfit(df[x_key], df[y_key], 1)
    plt.plot(df[x_key], m*df[x_key]+b, 'r:', alpha=0.2, linewidth=2)
    fig.suptitle(title)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    run[title].log(fig)
    fig.savefig(fname)
    plt.clf()
