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


tokenizer, model = get_model(train=True)

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

output_dir = os.path.join(config.MODEL_OUTPUT, config.MODEL_NAME)

# Epochs set to a lower number to avoid overfitting
model.fit(train_dataset.shuffle(101).batch(config.BATCH_SIZE), epochs=2,
          validation_data=val_dataset.batch(config.BATCH_SIZE))

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
