import argparse
import re
import numpy as np
import transformers
import tensorflow as tf
import pandas as pd

from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

from feature_engineering.song_loader import get_song_df
from .model_assembler import create_direct_model
from transformers import DistilBertTokenizer
from .tf_configurator import get_num_gpus, init_neptune, tf_config

distil_bert = 'distilbert-base-uncased'

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

    X, y = song_df.drop(['valence', 'arousal'], axis=1), song_df[['valence', 'arousal']]
    X = generate_embeddings(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y.values, train_size=0.8)

    # TODO - find optimal token length
    # ds = DiscourseDataSet(song_df,
    #                       num_labels=2,
    #                       seq_len=128,
    #                       test_prop=0.15,
    #                       batch_size=(64 * get_num_gpus()),
    #                       options=ds_options)

    with distribution_strategy.scope():
        model = create_direct_model()
        load_weights(model, args.model)
        print(model.summary())

        inputs = [np.asarray(x).astype('int') for x in [X_train['input_token'], X_train['masked_token']]]
        model.fit(inputs=inputs, y=y_train, verbose=1, batch_size=(64 * get_num_gpus()), callbacks=callbacks,
                  epochs=args.num_epoch)
        model.save_weights('r_amg_model_finished')

        # print("\n\nValidating...")
        # model.evaluate(ds.validate, verbose=1, callbacks=callbacks)

        print("\n\nTesting...")
        # TODO

        preds = model.predict(X_test, verbose=1, callbacks=callbacks)
        print(preds)
        valence_corr = pearsonr(y_test[[0]], preds[[0]])
        arr_corr = pearsonr(y_test[[1]], preds[[1]])
        print(f"Pearson's Correlation - Valence: {valence_corr}")
        print(f"Pearson's Correlation - Valence: {arr_corr}")


def tokenize(comments: pd.Series, tokenizer) -> transformers.BatchEncoding:
    return tokenizer(list(comments),
                     add_special_tokens=True,
                     return_attention_mask=True,
                     return_token_type_ids=False,
                     max_length=128,
                     padding='max_length',
                     truncation=True,
                     return_tensors='np')


def generate_embeddings(df: pd.DataFrame) -> tf.data.Dataset:
    tokenizer = DistilBertTokenizer.from_pretrained(distil_bert,
                                                    do_lower_case=True,
                                                    add_special_tokens=True,
                                                    max_length=128,
                                                    padding='max_length',
                                                    truncate=True,
                                                    padding_side='right')

    encodings = tokenize(df['body'], tokenizer)
    df['input_token'] = encodings['input_ids']
    df['masked_token'] = encodings['attention_mask']
    return df
