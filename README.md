# GPT2 Finetuned Sentiment Analysis Model

This model is fine-tuned on the **Tweet Sentiment Extraction** dataset for sentiment analysis. It classifies tweets into three sentiment categories: Positive, Negative, and Neutral.

## Model Description

- **Model Type**: GPT-2 for sequence classification
- **Task**: Sentiment Analysis
- **Dataset**: [Tweet Sentiment Extraction](https://huggingface.co/datasets/mteb/tweet_sentiment_extraction)
- **Training Objective**: Classify tweets into one of the three sentiment categories: Positive, Negative, or Neutral.

## Model Link on Hugging Face

You can find the model here: [Sentiment Analysis Model on Hugging Face](https://huggingface.co/riturajpandey739/gpt2-sentiment-analysis-tweets)

## Intended Use

This model can be used to classify tweets based on sentiment. It is specifically trained on tweets, but could be used for other short text sentiment analysis tasks with fine-tuning.

## Model Details

- **Architecture**: GPT-2 fine-tuned for classification
- **Input**: Text (Tweet)
- **Output**: Integer (0 for Negative, 1 for Neutral, 2 for Positive)
- **Training Data**: The model was trained using the Tweet Sentiment Extraction dataset available on the Hugging Face datasets hub.
- **Fine-tuning**: Fine-tuned for sentiment classification with 1000 examples from the training set.

## Performance

The model was evaluated using accuracy metrics on a validation set, achieving an accuracy of **XX%**.

## Limitations and Biases

- The model is primarily trained on tweets in English, so its performance may not be as strong on non-English text or highly specialized language.
- As with any sentiment analysis model, it may have difficulty with sarcasm, irony, and highly context-dependent sentiment.

## How to Use

To use the model with the Hugging Face Transformers library, follow these steps:

```python
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch

# Load the model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("riturajpandey739/gpt2-sentiment-analysis-tweets")
model = GPT2ForSequenceClassification.from_pretrained("riturajpandey739/gpt2-sentiment-analysis-tweets")

# Tokenize the input text
input_text = "This is a fantastic product! I highly recommend it."
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# Get model predictions
with torch.no_grad():
    logits = model(**inputs).logits

# Get the predicted class (0, 1, 2 for Negative, Neutral, Positive)
predicted_class = torch.argmax(logits, dim=-1).item()

print(f"Predicted Label: {predicted_class}")
