import unittest
import unittest.mock

import requests
import json


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

    response_data = json.loads(response.text)
    sentiment = response_data["sentiment"]
    score = response_data["score"]
    print("Sentiment: ", sentiment)
    print("Score: ", score)

    return sentiment, score


class TestCloudFunction(unittest.TestCase):
    def test_sentiment(request):

        sentences = {
            "Who are you": "Positive",
            "Get lost!": "Negative"
        }

        for sentence, sentiment in sentences.items():
            pred_sentiment, _ = req_and_resp(sentence)
            assert pred_sentiment == sentiment
