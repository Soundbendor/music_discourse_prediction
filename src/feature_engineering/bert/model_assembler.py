import tensorflow as tf

from transformers import TFDistilBertForSequenceClassification
from transformers import DistilBertConfig


distil_bert = 'distilbert-base-uncased'
NUM_LABEL = 2
MAX_SEQ_LEN = 128


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
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    # model.get_layer(name='tf_distil_bert_for_sequence_classification').trainable = False
    return model
