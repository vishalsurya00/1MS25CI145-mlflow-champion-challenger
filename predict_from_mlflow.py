
import mlflow
import pandas as pd

MODEL_URI = "models:/Sentiment_Classifier@champion"

# Load the Champion directly from MLflow
print("Loading Champion model from MLflow...")

model = mlflow.pyfunc.load_model(MODEL_URI)

from mlflow import MlflowClient

client = MlflowClient()

model_version = client.get_model_version_by_alias(
    "Sentiment_Classifier",
    "champion"
)

print("Registered model version:", model_version.version)
print("Hugging Face model ID:", "textattack/bert-base-uncased-SST-2")

print("Champion model loaded successfully!")

# New, unseen sentences
new_sentences = [
    "This movie was absolutely fantastic!",
    "I hated every minute of this film.",
    "The acting was brilliant and the story was beautiful.",
    "The movie was boring and disappointing.",
    "I would definitely recommend this movie.",
    "The worst movie I have ever watched.",
    "The performances were excellent.",
    "I regret spending my time watching this.",
    "A wonderful and entertaining experience.",
    "Nothing about this movie impressed me."
]

# Convert sentences into a DataFrame
input_data = pd.DataFrame({
    "sentence": new_sentences
})

# Generate predictions
predictions = model.predict(input_data)

# Combine sentences and predictions
results = pd.concat(
    [input_data, predictions],
    axis=1
)

print("\nNew Data Predictions:")
print(results.to_string(index=False))

# Save predictions
results.to_csv(
    "outputs/new_data_predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")
print("File: outputs/new_data_predictions.csv")
