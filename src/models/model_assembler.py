import tensorflow as tf
from transformers import (AutoConfig, TFAutoModel,
                          TFAutoModelForSequenceClassification)

NUM_LABEL = 2
MAX_SEQ_LEN = 128


def create_model(model_name: str) -> tf.keras.Model:
    config = AutoConfig.from_pretrained(model_name)
    db_seq = TFAutoModel.from_config(config)

    # for layer in db_seq.layers:
    #     layer.trainable = False
    #     for w in layer.weights:
    #         w._trainable = False

    print(db_seq)

    input_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), name="input_token", dtype="int32")
    input_masks_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), name="masked_token", dtype="int32")

    output = db_seq(input_ids, input_masks_ids).last_hidden_state[:, 0, :]
    # output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(MAX_SEQ_LEN)(output)
    output = tf.keras.layers.LeakyReLU(alpha=0.2)(output)
    # output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dense(2, activation="linear")(output)

    model = tf.keras.Model(inputs=[input_ids, input_masks_ids], outputs=output)
    opt = tf.keras.optimizers.Adam(learning_rate=5e-5)

    model.compile(optimizer=opt, loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model
