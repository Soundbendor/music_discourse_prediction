import transformers
import tensorflow as tf
import pandas as pd

from transformers import TFDistilBertModel
from transformers import DistilBertConfig
from transformers import DistilBertTokenizer


distil_bert = 'distilbert-base-uncased'
NUM_LABEL = 2
MAX_SEQ_LEN = 128


def _distilbert_layer(config: DistilBertConfig, input_ids, mask_ids) -> tf.keras.layers.Layer:
    config.output_hidden_states = False
    transformer_model = TFDistilBertModel.from_pretrained(distil_bert, config = config)
    return transformer_model(input_ids, mask_ids)[0]


def create_model() -> tf.keras.Model:
    config = DistilBertConfig(dropout=0.2, attention_dropout=0.2)
    
    input_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), name='input_token', dtype='int32')
    input_masks_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), name='masked_token', dtype='int32')

    embed_layer = _distilbert_layer(config, input_ids, input_masks_ids)
    output = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True, dropout=0.1))(embed_layer)
    # TODO - compare to BertPooler
    # TODO - possibly try with 
    output = tf.keras.layers.GlobalMaxPool1D()(output)
    output = tf.keras.layers.Dense(50, activation='relu')(output)
    output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(NUM_LABEL, activation='relu')(output)
    model = tf.keras.Model(inputs=[input_ids, input_masks_ids], outputs = output)

    opt = tf.keras.optimizers.Adam(learning_rate=5e-5)

    model.compile(optimizer=opt, loss=tf.keras.losses.CosineSimilarity(axis=1), metrics=tf.keras.metrics.RootMeanSquaredError())
    model.get_layer(name='tf_distil_bert_model').trainable = False
    return model

def _tokenize(comments: pd.Series, tokenizer) -> transformers.BatchEncoding:
    return tokenizer(list(comments), add_special_tokens=True, return_attention_mask=True,
                    return_token_type_ids=False, max_length=MAX_SEQ_LEN, padding='max_length',
                    truncation=True, return_tensors='tf')


def generate_embeddings(df: pd.DataFrame) -> tf.data.Dataset:
    # Initialize tokenizer - set to automatically lower-case
    tokenizer = DistilBertTokenizer.from_pretrained(distil_bert,
        do_lower_case=True, add_special_tokens=True, max_length=MAX_SEQ_LEN, padding='max_length', truncate=True, padding_side='right')
    encodings = _tokenize(df['body'], tokenizer)
    return tf.data.Dataset.from_tensor_slices(({
        'input_token': encodings['input_ids'],
        'masked_token': encodings['attention_mask'],
    }, tf.constant((df[['valence', 'arousal']].values).astype('float32')))) 
