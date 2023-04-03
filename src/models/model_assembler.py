import tensorflow as tf

from transformers.utils.dummy_tf_objects import TFAutoModel


NUM_LABEL = 2
MAX_SEQ_LEN = 128


def create_model(model_name: str) -> tf.keras.Model:
    # config = DistilBertConfig(num_labels=NUM_LABEL)
    db_seq = TFAutoModel.from_pretrained(model_name)
    print(db_seq)

    input_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), name="input_token", dtype="int32")
    input_masks_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), name="masked_token", dtype="int32")

    output = db_seq(input_ids, input_masks_ids).last_hidden_state[:, 0, :]
    # output = tf.keras.layers.Dense(768, activation='relu')(output)
    output = tf.keras.layers.Dense(MAX_SEQ_LEN, activation="relu")(output)
    output = tf.keras.layers.Dense(2, activation="relu")(output)

    model = tf.keras.Model(inputs=[input_ids, input_masks_ids], outputs=output)
    opt = tf.keras.optimizers.Adam(learning_rate=5e-5)

    model.compile(
        optimizer=opt, loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError(), correlation_coefficient_loss]
    )
    # model.get_layer(index=2).trainable = False
    return model
