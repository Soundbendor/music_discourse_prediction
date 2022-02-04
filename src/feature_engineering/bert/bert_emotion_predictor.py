import argparse
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers

from typing import List, Tuple
from tqdm import tqdm
from transformers import DistilBertTokenizer
from transformers import TFDistilBertModel
from transformers import DistilBertConfig

from feature_engineering.song_loader import get_song_df

# - load data from json 
# - remove url's, html tags (maybe use that regex from wordbag?)
# - tokenize
# - convert attn. mask, input IDs, and label into tensor dataset
# - build model architecture
# -   -   adam optimizer with 5e-5 learn rate 
# -   -   distilBERT with regression head on top (yes, i know it's called classification)
# -   -    -    since it's a multi-target regression task, uses cross-entropy as loss function

distil_bert = 'distilbert-base-uncased'
MAX_SEQ_LEN = 512
NUM_LABEL = 2

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

def tokenize(comments: pd.Series, tokenizer) -> transformers.BatchEncoding:
    return tokenizer(list(comments), add_special_tokens=True,
        return_attention_mask=True, return_token_type_ids=False, max_length=MAX_SEQ_LEN, padding='max_length', truncation=True, return_tensors='tf')

    
def generate_embeddings(df: pd.DataFrame, tokenizer) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    encodings = tokenize(df['body'], tokenizer)
    
    return np.asarray(encodings['input_ids']), np.asarray(encodings['attention_mask']), df[['valence', 'arousal']].values).astype('float32')))

def main():
    args = parseargs()
    tqdm.pandas()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # tf.debugging.set_log_device_placement(True)

    song_df = get_song_df(args.input)

    # Clean strings - remove urls and html tags
    rx = re.compile(r'(?:<.*?>)|(?:http\S+)')
    song_df['body'] = song_df['body'].apply(lambda x: rx.sub('', x))

    # Initialize tokenizer - set to automatically lower-case
    tokenizer = DistilBertTokenizer.from_pretrained(distil_bert,
        do_lower_case=True, add_special_tokens=True, max_length=MAX_SEQ_LEN, padding='max_length', truncate=True)

    # Create tf.Dataset with input ids, attention mask, and [valence, arousal] target labels
    # TODO - need a train-test split
    ids, attention_mask, labels = generate_embeddings(song_df, tokenizer)


    config = DistilBertConfig(dropout=0.2, attention_dropout=0.2)
    config.output_hidden_states = False
    transformer_model = TFDistilBertModel.from_pretrained(distil_bert, config = config)
    
    input_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), name='input_token', dtype='int32')
    input_masks_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), name='masked_token', dtype='int32')

    embed_layer = transformer_model(input_ids, input_masks_ids)[0]
    output = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(embed_layer)
    output = tf.keras.layers.GlobalAveragePooling1D()(output)
    output = tf.keras.layers.Dense(50, activation='relu')(output)
    output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(NUM_LABEL, activation='relu')(output)
    model = tf.keras.Model(inputs=[input_ids, input_masks_ids], outputs = output)

    opt = tf.keras.optimizers.Adam(learning_rate=5e-5)

    model.compile(optimizer=opt, loss=tf.keras.losses.CosineSimilarity(axis=1), metrics=tf.keras.metrics.RootMeanSquaredError())

    model.get_layer(name='tf_distil_bert_model').trainable = False

    print(model.summary())
    # print(list(song_embeddings.take(1).as_numpy_iterator()))


    # TODO - neptune
    model.fit(X={'input_token': ids, 'masked_token': attention_mask}, y = labels verbose=1, epochs=100)

    # logits = model.predict([song_df['input_ids'].to_numpy(), song_df['input_masks'].to_numpy()], verbose=1).logits

    
    
