import tensorflow as tf
from transformers import (AutoConfig, TFAutoModel,
                          TFAutoModelForSequenceClassification)

NUM_LABEL = 2
MAX_SEQ_LEN = 64


def create_model(model_name: str) -> tf.keras.Model:
    # config = DistilBertConfig(num_labels=NUM_LABEL)
    config = AutoConfig.from_pretrained(model_name)
    print(model_name)
    db_seq = TFAutoModel.from_config(config)

    for layer in db_seq.layers:
        layer.trainable = False
        for w in layer.weights:
            w._trainable = False

    print(db_seq)

    input_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), name="input_token", dtype="int32")
    input_masks_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), name="masked_token", dtype="int32")

    output = db_seq(input_ids, input_masks_ids).last_hidden_state[:, 0, :]
    output = tf.keras.layers.Dropout(0.3)(output)
    output = tf.keras.layers.Dense(MAX_SEQ_LEN, activation="relu")(output)
    output = tf.keras.layers.Dense(2, activation="linear")(output)

    model = tf.keras.Model(inputs=[input_ids, input_masks_ids], outputs=output)
    opt = tf.keras.optimizers.Adam(learning_rate=5e-5)

    model.compile(optimizer=opt, loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


def create_classification_model(model_name: str) -> tf.keras.Model:
    config = AutoConfig.from_pretrained(model_name, num_labels=64)
    db_seq = TFAutoModelForSequenceClassification.from_config(config)
    db_seq.layers[0].trainable = False

    print(db_seq)

    input_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), name="input_token", dtype="int32")
    input_masks_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), name="masked_token", dtype="int32")

    output = db_seq(input_ids, input_masks_ids, output_hidden_states=True).logits
    output = tf.keras.layers.Dense(32, activation="relu")(output)
    output = tf.keras.layers.Dense(2, activation="linear")(output)

    model = tf.keras.Model(inputs=[input_ids, input_masks_ids], outputs=output)
    opt = tf.keras.optimizers.Adam(learning_rate=5e-5)

    # # TODO - Won't work with auto models!
    # for layer in model.layers:
    #     print(layer)

    # for w in model.get_layer("tf_distil_bert_model").weights:
    #     w._trainable = False

    model.compile(optimizer=opt, loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model
