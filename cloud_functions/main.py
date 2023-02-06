import os
import flask
import numpy as np
import tensorflow as tf

from google.cloud import storage

from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from transformers import logging

# Set logging verbosity to show errors and not warnings
logging.set_verbosity_error()

BUCKET = os.environ.get('BUCKET').lower()
VERSION = os.environ.get('VERSION')

MODEL_FILES = [
    "config.json",
    "special_tokens_map.json",
    "tf_model.h5",
    "tokenizer_config.json",
    "vocab.txt"
]

MODEL_PATH = os.path.join("models", "distilbert-base-uncased")

load_path = os.path.join('/tmp', MODEL_PATH)

if not os.path.exists(load_path):
    os.makedirs(load_path)


def get_classifier_output(tokenizer, model, sentence):
    tokens = tokenizer(
        sentence,
        truncation=True,
        max_length=128,
        padding='max_length',
        return_tensors="tf"
    )

    output = model(tokens)
    tf_predictions = tf.nn.softmax(output[0], axis=-1)

    classes = np.argmax(tf_predictions.numpy(), axis=-1)

    return classes


def download_blob(bucket_name, source_blob_name, destination_file_name):

    storage_client = storage.Client()

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))


tokenizer = None
model = None


def sentiment(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <http://flask.pocoo.org/docs/1.0/api/#flask.Request>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>.
    """
    json_data = request.get_json()
    print(json_data)
    global tokenizer, model

    if json_data and "text" in json_data:
        text = json_data["text"]
    else:
        raise ValueError('Missing Field: text')

    if model is None:
        for file in MODEL_FILES:
            download_blob(BUCKET, os.path.join(
                'sentiment', MODEL_PATH, file), os.path.join('/tmp', MODEL_PATH, file))

        tokenizer = DistilBertTokenizer.from_pretrained(load_path)
        model = TFDistilBertForSequenceClassification.from_pretrained(
            load_path)

    model_input = text
    sentiment = get_classifier_output(
        tokenizer, model, model_input)
    sentiment = sentiment.item()

    if sentiment == 1:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'

    return sentiment
