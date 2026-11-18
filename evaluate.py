
import time
import pandas as pd
import mlflow

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt
import os

from config import (
    MODEL_A_NAME,
    MODEL_B_NAME,
    DATASET_PATH,
    EXPERIMENT_NAME
)

from hf_wrapper import SentimentModel

def save_confusion_matrix(actual_labels, predicted_labels, model_name):

    cm = confusion_matrix(actual_labels, predicted_labels)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["NEGATIVE", "POSITIVE"]
    )

    display.plot(cmap="Blues", values_format="d")

    plt.title(f"Confusion Matrix - {model_name}")

    os.makedirs("outputs", exist_ok=True)

    file_path = f"outputs/{model_name}_confusion_matrix.png"

    plt.savefig(file_path, dpi=300, bbox_inches="tight")

    plt.close()

    print(f"Confusion matrix saved: {file_path}")

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

    save_confusion_matrix(
    actual_labels,
    predicted_labels,
    run_name.replace(" ", "_").replace("-", "").lower()
)
    mlflow.log_artifact(
    f"outputs/{run_name.replace(' ', '_').replace('-', '').lower()}_confusion_matrix.png"
)

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

def save_comparison_chart(results_a, results_b):

    metrics = ["accuracy", "precision", "recall", "f1_score"]

    model_a_values = [results_a[m] for m in metrics]
    model_b_values = [results_b[m] for m in metrics]

    x = range(len(metrics))
    width = 0.35

    plt.figure(figsize=(9, 5))

    plt.bar(
        [i - width / 2 for i in x],
        model_a_values,
        width,
        label="Model A - DistilBERT"
    )

    plt.bar(
        [i + width / 2 for i in x],
        model_b_values,
        width,
        label="Model B - BERT"
    )

    plt.xticks(list(x), metrics)
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.title("Champion vs Challenger - Performance Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    os.makedirs("outputs", exist_ok=True)

    file_path = "outputs/model_comparison.png"

    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Comparison chart saved: {file_path}")

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

        # Generate the model comparison chart
    save_comparison_chart(results_a, results_b)

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

                # Log comparison chart in MLflow
        mlflow.log_artifact("outputs/model_comparison.png")

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
