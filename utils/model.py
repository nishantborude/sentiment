import os
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

from config import config


def get_model(train=False):
    if train:
        tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
        model = TFDistilBertForSequenceClassification.from_pretrained(
            config.MODEL_NAME, num_labels=config.NUM_CLASSES)
    else:
        model_dir = os.path.join(config.MODEL_OUTPUT, config.MODEL_NAME)
        tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        model = TFDistilBertForSequenceClassification.from_pretrained(
            model_dir)

    # Learning rate tweaked based on experiments
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return tokenizer, model
