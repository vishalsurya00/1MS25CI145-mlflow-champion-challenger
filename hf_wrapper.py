
from transformers import pipeline


class SentimentModel:

    def __init__(self, model_name):
        self.model_name = model_name

        print(f"\nLoading model: {model_name}")

        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=-1
        )

        print("Model loaded successfully!")

    def predict(self, sentences):
        results = self.classifier(
            sentences,
            batch_size=16,
            truncation=True
        )

        predictions = []

        for result in results:
            label = result["label"].upper()

            # Convert model output into a standard format
            if label in ["LABEL_1", "POSITIVE"]:
                sentiment = "POSITIVE"
            else:
                sentiment = "NEGATIVE"

            predictions.append({
                "sentiment": sentiment,
                "confidence": result["score"]
            })

        return predictions
