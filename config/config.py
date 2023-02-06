
# Path to relative location where data is stored
DATA_PATH = 'data/crypto_reddit_sentiment.csv'

# Model name to be used for training the sentiment classifier
MODEL_NAME = "distilbert-base-uncased"

# Positive and Negative sentiment
NUM_CLASSES = 2

# Directory to store images, if any
IMAGES = 'images'

# Model output directory to store model
MODEL_OUTPUT = 'models'

# Hyperparameter: Max Sentence Length (tweaked from experiments and EDA)
MAX_LENGTH = 128

# Batch Size
BATCH_SIZE = 32
