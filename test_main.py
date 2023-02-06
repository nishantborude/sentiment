import unittest
import unittest.mock

import requests


def structure_query(text):
    return {"text": text}


def req_and_resp(text):
    url = "https://us-central1-sentiment-http.cloudfunctions.net/sentiment_V1"

    query_data = structure_query(text)
    response = requests.post(
        "{}/".format(url), json=query_data
    )
    try:
        assert response.status_code == 200
    except Exception:
        print(response.status_code)
        print(response.text)
        return

    sentiment = response.text
    print("Sentiment: ", sentiment)

    return sentiment


class TestCloudFunction(unittest.TestCase):
    def test_sentiment(request):

        sentences = {
            "Who are you": "Positive",
            "Get lost!": "Negative"
        }

        for sentence, sentiment in sentences.items():

            assert req_and_resp(sentence) == sentiment
