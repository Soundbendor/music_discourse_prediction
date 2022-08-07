import tensorflow as tf
import tensorflow.keras.backend as K

from transformers import TFDistilBertForSequenceClassification
from transformers import TFDistilBertModel
from transformers import DistilBertConfig
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoConfig


distil_bert = 'distilbert-base-uncased'
roberta = 'cardiffnlp/twitter-roberta-base-emotion'
NUM_LABEL = 2
MAX_SEQ_LEN = 128


def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)


def create_model() -> tf.keras.Model:
    config = DistilBertConfig(num_labels=NUM_LABEL)
    db_seq = TFDistilBertForSequenceClassification.from_pretrained(
        distil_bert, config=config)

    input_ids = tf.keras.layers.Input(
        shape=(MAX_SEQ_LEN,), name='input_token', dtype='int32')
    input_masks_ids = tf.keras.layers.Input(
        shape=(MAX_SEQ_LEN,), name='masked_token', dtype='int32')

    output = db_seq(input_ids, input_masks_ids)[0]
    # output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(2, activation='relu')(output)

    model = tf.keras.Model(inputs=[input_ids, input_masks_ids], outputs=output)
    opt = tf.keras.optimizers.Adam(learning_rate=5e-5)

    model.compile(
        optimizer=opt,
        loss='mse',
        metrics=[tf.keras.metrics.RootMeanSquaredError(), correlation_coefficient_loss]
    )
    # model.get_layer(name='tf_distil_bert_for_sequence_classification').trainable = False
    return model


def create_model_new() -> tf.keras.Model:
    # config = DistilBertConfig(num_labels=NUM_LABEL)
    db_seq = TFDistilBertModel.from_pretrained(distil_bert)

    input_ids = tf.keras.layers.Input(
        shape=(MAX_SEQ_LEN,), name='input_token', dtype='int32')
    input_masks_ids = tf.keras.layers.Input(
        shape=(MAX_SEQ_LEN,), name='masked_token', dtype='int32')

    output = db_seq(input_ids, input_masks_ids)[0]
    output = tf.keras.layers.Dense(MAX_SEQ_LEN, activation='relu')(output)
    output = tf.keras.layers.Dense(2, activation='relu')(output)

    model = tf.keras.Model(inputs=[input_ids, input_masks_ids], outputs=output)
    opt = tf.keras.optimizers.Adam(learning_rate=5e-5)

    model.compile(
        optimizer=opt,
        loss='mse',
        metrics=[tf.keras.metrics.RootMeanSquaredError(), correlation_coefficient_loss]
    )
    # model.get_layer(name='tf_distil_bert_for_sequence_classification').trainable = False
    return model


def create_roberta_model() -> tf.keras.Model:
    config = AutoConfig(num_labels=NUM_LABEL)
    db_seq = TFAutoModelForSequenceClassification.from_pretrained(
        roberta, config=config)

    input_ids = tf.keras.layers.Input(
        shape=(MAX_SEQ_LEN,), name='input_token', dtype='int32')
    input_masks_ids = tf.keras.layers.Input(
        shape=(MAX_SEQ_LEN,), name='masked_token', dtype='int32')

    output = db_seq(input_ids, input_masks_ids)[0]
    # output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(2, activation='relu')(output)

    model = tf.keras.Model(inputs=[input_ids, input_masks_ids], outputs=output)
    opt = tf.keras.optimizers.Adam(learning_rate=5e-5)

    model.compile(
        optimizer=opt,
        loss='mse',
        metrics=[tf.keras.metrics.RootMeanSquaredError(), correlation_coefficient_loss]
    )
    # model.get_layer(name='tf_distil_bert_for_sequence_classification').trainable = False
    return model