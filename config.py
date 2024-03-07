from transformers import pipeline
DATA_PATH = "sentiment_analysis_data.txt"
# Assuming you want to use a specific model, adjust the device, and manage memory more efficiently
model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Example model
device = -1  # -1 for CPU, CUDA device index (e.g., 0) for GPU
return_all_scores = True  # To get scores for all categories (e.g., positive and negative)

# Load the sentiment analysis pipeline with additional parameters
sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, device=device)