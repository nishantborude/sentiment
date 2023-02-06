# Sentiment Classifier and deployment
This presents code for PoC Sentiment Classifier model and deploying as Cloud Function to serve certain requirements.

## Dataset

The dataset is split into 80-20, making sure the relative proportion is maintained in train and test sets. 

### Preprocessing

Since the data contains texts of varying lengths, I took the histogram to count the number of sentences across the varying lengths and it turned out majority of the data was under 200 words per sentence. To get decent training and testing errors, the hyperparameter I chose was 128. All the sentences are either truncated to 128 if greater than that or padded if less than that.

## Model

I fine-tuned a pretrained DistillBERT model on the Crypto Sentiment data. This model is smaller and will easily run on the cloud function and will help us run inference quickly while ensuring minimum costs for large number of inferences.

The model would not upload on github hence it is stored on GCS at:
https://console.cloud.google.com/storage/browser/models_store_sentiment/sentiment?project=sentiment-http


## Training

The training here is straightforward with Adam optimizer with learning rate between 1e-4 and 1e-6. The loss function is Sparse Categorical Cross Entropy. One thing to note is that is a general N-class implementation, using just 1 output class and Binary Cross Entropy loss will also work. I trained only for 2-3 epochs across different experiments to avoid overfitting.

## Testing

The test or val accuracy for this model is 86% and can be improved by additional hyperparameter and fine-tuning. The goal here is to deploy a working prototype.

## Estimated costs

According to: https://cloud.google.com/functions/pricing, 

The cost for:
1 million invocations = $0.40
so for 10 million invocations = $4 
For handling 10 million requests per day, monthly cost = 30 * 4 = $120 
Adding networking costs and for high volume requests such as:
50,000,000 invocations x (5 KB / 1024 KB/MB / 1024 MB/GB) = 238.42 GB of egress traffic per month it comes to about $160 per month.

This can be easily reduced by model optimizing methods such as quantization to reduce the memory footprint thus halving the costs.



