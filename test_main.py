import unittest
import unittest.mock

from cloud_functions.main import sentiment


class TestCloudFunction(unittest.TestCase):
    def test_cloud_function(self):
        text = 'Hi There!'
        req = unittest.mock.Mock(args={"text": text})

        # Call tested function
        print(sentiment(req))
        assert sentiment(req) == 1
