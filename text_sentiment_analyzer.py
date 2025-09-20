# 3_text_sentiment_analyzer.py
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class TextSentimentAnalyzer:
    def __init__(self):
        self.MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.LOCAL_MODEL_PATH = os.path.join(os.getcwd(), "local_roberta_sentiment_model")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Initializing Text Sentiment Analyzer...")
        self.tokenizer, self.model = self._load_model()
        self.model.eval()
        print("Text Sentiment Analyzer initialized.")

    def _load_model(self):
        if os.path.exists(self.LOCAL_MODEL_PATH):
            print(f"Loading text model from local path: {self.LOCAL_MODEL_PATH}")
            tokenizer = AutoTokenizer.from_pretrained(self.LOCAL_MODEL_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(self.LOCAL_MODEL_PATH).to(self.device)
        else:
            print("Downloading text model (one-time operation)...")
            tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME).to(self.device)
            tokenizer.save_pretrained(self.LOCAL_MODEL_PATH)
            model.save_pretrained(self.LOCAL_MODEL_PATH)
            print(f"Text model saved to {self.LOCAL_MODEL_PATH}")
        return tokenizer, model

    def analyze_text(self, text: str) -> str:
        """
        Analyzes a text string for sentiment.

        Args:
            text: The user's input text.

        Returns:
            The detected sentiment label (str).
        """
        if not text or not text.strip():
            return "N/A"

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
        sentiment = self.model.config.id2label[predicted_class_id]
        
        # We need to map this to our common emotion set
        if sentiment == 'positive':
            return 'Happy'
        elif sentiment == 'negative':
            return 'Sad' # Or 'Angry' - this is a simplification
        else:
            return 'Neutral'

if __name__ == '__main__':
    # This block allows you to test this file independently
    print("Testing Text Sentiment Analyzer...")
    analyzer = TextSentimentAnalyzer()
    test_text = "The view from the cupola is breathtaking."
    result = analyzer.analyze_text(test_text)
    print(f"Text: '{test_text}'\nSentiment: {result}")