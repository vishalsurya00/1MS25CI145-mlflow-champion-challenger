
import pandas as pd
import mlflow
from mlflow import MlflowClient

from config import MODEL_A_NAME, MODEL_B_NAME, EXPERIMENT_NAME
from hf_wrapper import SentimentModel


REGISTERED_MODEL_NAME = "Sentiment_Classifier"


class HuggingFacePyFunc(mlflow.pyfunc.PythonModel):

    def __init__(self, model_name):
        self.model_name = model_name

    def load_context(self, context):
        self.model = SentimentModel(self.model_name)

    def predict(self, context, model_input, params=None):

        if isinstance(model_input, pd.DataFrame):
            sentences = model_input["sentence"].astype(str).tolist()
        else:
            sentences = list(model_input)

        predictions = self.model.predict(sentences)

        return pd.DataFrame(predictions)


def register_model(model_name, alias):

    print(f"\nRegistering: {model_name}")
    print(f"Alias: {alias}")

    with mlflow.start_run(run_name=f"register_{alias}"):

        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=HuggingFacePyFunc(model_name),
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=pd.DataFrame({
                "sentence": [
                    "This movie is amazing!"
                ]
            }),
            pip_requirements=[
                "mlflow==3.18.0",
                "transformers",
                "torch",
                "pandas"
            ]
        )

    version = model_info.registered_model_version

    client = MlflowClient()

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=alias,
        version=str(version)
    )

    print("Registration successful!")
    print("Model name:", REGISTERED_MODEL_NAME)
    print("Version:", version)
    print("Alias:", alias)


def main():

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Model B is the winner
    register_model(MODEL_B_NAME, "champion")

    # Model A is the challenger
    register_model(MODEL_A_NAME, "challenger")

    print("\nBoth models registered successfully!")


if __name__ == "__main__":
    main()
