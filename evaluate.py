
import time
import pandas as pd
import mlflow

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from config import (
    MODEL_A_NAME,
    MODEL_B_NAME,
    DATASET_PATH,
    EXPERIMENT_NAME
)

from hf_wrapper import SentimentModel


def evaluate_model(model_name, run_name, sentences, actual_labels):

    print(f"\nEvaluating: {run_name}")

    model = SentimentModel(model_name)

    # Warm-up prediction (not included in timing)
    model.predict(["This is a warm-up sentence."])

    # Measure prediction time
    start_time = time.perf_counter()

    predictions = model.predict(sentences)

    total_time = time.perf_counter() - start_time

    predicted_labels = [
        1 if item["sentiment"] == "POSITIVE" else 0
        for item in predictions
    ]

    # Calculate performance metrics
    accuracy = accuracy_score(actual_labels, predicted_labels)
    precision = precision_score(actual_labels, predicted_labels, zero_division=0)
    recall = recall_score(actual_labels, predicted_labels, zero_division=0)
    f1 = f1_score(actual_labels, predicted_labels, zero_division=0)

    average_latency = total_time / len(sentences)

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "total_prediction_time": total_time,
        "average_latency": average_latency
    }

    print(f"\nResults for {run_name}:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    return results


def main():

    # Load the fixed evaluation dataset
    df = pd.read_csv(DATASET_PATH)

    sentences = df["sentence"].tolist()
    actual_labels = df["label"].tolist()

    print("Total evaluation sentences:", len(sentences))

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Evaluate Model A
    with mlflow.start_run(run_name="A_sentiment_model"):
        mlflow.log_param("model_name", MODEL_A_NAME)
        results_a = evaluate_model(
            MODEL_A_NAME,
            "Model A - Champion",
            sentences,
            actual_labels
        )
        mlflow.log_metrics(results_a)

    # Evaluate Model B
    with mlflow.start_run(run_name="B_sentiment_model"):
        mlflow.log_param("model_name", MODEL_B_NAME)
        results_b = evaluate_model(
            MODEL_B_NAME,
            "Model B - Challenger",
            sentences,
            actual_labels
        )
        mlflow.log_metrics(results_b)

    
    # Compare both models and record the decision
    f1_improvement = results_b["f1_score"] - results_a["f1_score"]

    if f1_improvement >= 0.01:
        winner = "Model B"
        reason = "Challenger improved F1-score by at least 0.01"
    else:
        winner = "Model A"
        reason = "Challenger did not meet the minimum F1 improvement"

    with mlflow.start_run(run_name="compare_sentiment"):

        mlflow.log_param("champion_model", MODEL_A_NAME)
        mlflow.log_param("challenger_model", MODEL_B_NAME)
        mlflow.log_param("selection_rule", "F1 improvement >= 0.01")
        mlflow.log_param("selected_winner", winner)
        mlflow.log_param("decision_reason", reason)

        mlflow.log_metric("f1_improvement", f1_improvement)

        for metric, value in results_a.items():
            mlflow.log_metric(f"model_a_{metric}", value)

        for metric, value in results_b.items():
            mlflow.log_metric(f"model_b_{metric}", value)

    print("\nComparison completed!")
    print("F1 improvement:", round(f1_improvement, 4))
    print("Selected winner:", winner)
    print("Reason:", reason)



if __name__ == "__main__":
    main()
