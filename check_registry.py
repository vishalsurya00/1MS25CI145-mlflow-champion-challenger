
from mlflow import MlflowClient

client = MlflowClient()

for alias in ["champion", "challenger"]:

    version = client.get_model_version_by_alias(
        "Sentiment_Classifier",
        alias
    )

    print("\nAlias:", alias)
    print("Version:", version.version)
    print("Hugging Face ID:", version.tags.get("hf_model_id"))
