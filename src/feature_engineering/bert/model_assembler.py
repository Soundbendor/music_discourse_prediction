import tensorflow as tf

from transformers import TFDistilBertModel
from transformers import TFDistilBertForSequenceClassification
from transformers import DistilBertConfig


distil_bert = 'distilbert-base-uncased'
NUM_LABEL = 2
MAX_SEQ_LEN = 128


def _distilbert_layer(config: DistilBertConfig, input_ids, mask_ids) -> tf.keras.layers.Layer:
    config.output_hidden_states = False
    transformer_model = TFDistilBertModel.from_pretrained(distil_bert, config = config)
    # summoning undocumented bullshit
    return transformer_model.distilbert(input_ids, mask_ids)[0]


def create_model() -> tf.keras.Model:
    config = DistilBertConfig(dropout=0.2, attention_dropout=0.2)
    
    input_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), name='input_token', dtype='int32')
    input_masks_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), name='masked_token', dtype='int32')

    embed_layer = _distilbert_layer(config, input_ids, input_masks_ids)
    output = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True, dropout=0.1))(embed_layer)
    output = tf.keras.layers.GlobalMaxPool1D()(output)
    output = tf.keras.layers.Dense(50, activation='relu')(output)
    output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(NUM_LABEL, activation='relu')(output)
    model = tf.keras.Model(inputs=[input_ids, input_masks_ids], outputs = output)

    opt = tf.keras.optimizers.Adam(learning_rate=5e-5)

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CosineSimilarity(axis=1),
        metrics=tf.keras.metrics.RootMeanSquaredError()
    )
    model.get_layer(name='distilbert').trainable = False
    return model

def create_direct_model() -> tf.keras.Model:
    config = DistilBertConfig(num_labels=1)
    db_seq = TFDistilBertForSequenceClassification.from_pretrained(distil_bert, config=config)

    input_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), name='input_token', dtype='int32')
    input_masks_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), name='masked_token', dtype='int32')

    output = db_seq(input_ids, input_masks_ids)[0]
    output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(2, activation='relu')(output)

    model = tf.keras.Model(inputs=[input_ids, input_masks_ids], outputs = output)
    opt = tf.keras.optimizers.Adam(learning_rate=5e-5)

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CosineSimilarity(axis=1),
        metrics=tf.keras.metrics.RootMeanSquaredError()
    )
    model.get_layer(name='tf_distil_bert_for_sequence_classification').trainable = False
    return model


    






