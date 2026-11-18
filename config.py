
# MODEL CONFIGURATION

MODEL_A_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

MODEL_B_NAME = "textattack/bert-base-uncased-SST-2"

DATASET_PATH = "data/evaluation_data.csv"

PRIMARY_METRIC = "f1_score"

MINIMUM_F1_IMPROVEMENT = 0.01

EXPERIMENT_NAME = "Champion_Challenger_Sentiment"
