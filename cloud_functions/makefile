VERSION=V1

deploy_function_sentiment:
	gcloud functions deploy sentiment_$(VERSION) \
	--entry-point sentiment \
	--runtime python310 \
	--project sentiment-http \
	--set-env-vars BUCKET=models_store_sentiment,VERSION=$(VERSION) \
	--trigger-http \
  	--allow-unauthenticated \
  	--memory 512 \