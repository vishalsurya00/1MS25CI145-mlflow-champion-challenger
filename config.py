
import pandas as pd

# Load the SST-2 validation dataset
df = pd.read_csv("data/SST-2/dev.tsv", sep="\t")

# Select 100 negative and 100 positive sentences
negative = df[df["label"] == 0].sample(n=100, random_state=42)
positive = df[df["label"] == 1].sample(n=100, random_state=42)

# Combine and shuffle the selected sentences
evaluation_data = pd.concat([negative, positive])
evaluation_data = evaluation_data.sample(frac=1, random_state=42)

# Save the evaluation dataset
evaluation_data.to_csv("data/evaluation_data.csv", index=False)

print("Evaluation dataset created successfully!")
print("Total sentences:", len(evaluation_data))
print(evaluation_data["label"].value_counts().sort_index())
print(evaluation_data.head())



# --------------------------------
# MODEL CONFIGURATION
# --------------------------------

MODEL_A_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

MODEL_B_NAME = "textattack/bert-base-uncased-SST-2"

# Evaluation dataset
DATASET_PATH = "data/evaluation_data.csv"

# Champion selection rule
PRIMARY_METRIC = "f1_score"

MINIMUM_F1_IMPROVEMENT = 0.01

# MLflow experiment
EXPERIMENT_NAME = "Champion_Challenger_Sentiment"
