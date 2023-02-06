import os
import pandas as pd
import matplotlib.pyplot as plt

from config import config


def read_csv():
    df = pd.read_csv(config.DATA_PATH)
    return df


def analyze_text_lengths(df):
    length = df['Comment Text'].apply(lambda x: len(x))
    min_text_length = min(length)
    max_text_length = max(length)
    avg_text_length = length.mean()

    print('Minimum length: ', min_text_length)
    print('Maximum length: ', max_text_length)
    print('Average length: ',  avg_text_length)

    return plot_histogram(length)


def plot_histogram(length):
    count_object = plt.hist(length)
    plt.savefig(os.path.join(config.IMAGES, 'comment_length_histogram.png'))
    plt.close()
    return


def process_sentiment_data(df):

    comments = df['Comment Text']
    labels = df['Sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)

    return comments.to_numpy(), labels.to_numpy()
