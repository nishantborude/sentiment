import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

from config import config
from utils.model import get_model
from utils.data_processing import read_csv, process_sentiment_data

df = read_csv()
comments, labels = process_sentiment_data(df)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    comments, labels, stratify=labels, test_size=.2, random_state=1)


tokenizer, model = get_model(train=False)

train_encodings = tokenizer(
    list(train_texts), truncation=True, max_length=config.MAX_LENGTH, padding='max_length')
val_encodings = tokenizer(list(val_texts), truncation=True,
                          max_length=config.MAX_LENGTH, padding='max_length')

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
))
val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels
))

val_loss, val_acc = model.evaluate(val_dataset.batch(config.BATCH_SIZE))

print('Val Accuracy: ', val_acc * 100)
