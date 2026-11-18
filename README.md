# Champion–Challenger Model Comparison with Hugging Face and MLflow

## 1. Project Overview

This project implements a Champion–Challenger framework to compare two pretrained sentiment classification models using Hugging Face Transformers and MLflow.

Both models are evaluated on the same dataset using standard classification metrics and prediction timing. MLflow is used to track experiments, compare results, register models, and manage model aliases.

## 2. Models Used

| Role | Model | Hugging Face Model ID |
|---|---|---|
| Model A – Baseline | DistilBERT | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` |
| Model B – Challenger | BERT | `textattack/bert-base-uncased-SST-2` |

## 3. Dataset

- Dataset: Stanford Sentiment Treebank (SST-2)
- Source: GLUE benchmark
- Split used: Validation (`dev.tsv`)
- Evaluation samples: 200 sentences
- Class distribution: 100 negative and 100 positive sentences
- Evaluation file: `data/evaluation_data.csv`

Both models receive exactly the same 200 sentences in the same order.

## 4. Evaluation Metrics

The following metrics are used:

- Accuracy
- Precision
- Recall
- F1-score
- Total prediction time
- Average prediction latency

## 5. Champion Selection Rule

F1-score is the primary selection metric.

The Challenger (Model B) is selected as the new Champion only if its F1-score improves over Model A by at least **0.01 (absolute improvement)**.

Otherwise, Model A remains the Champion.

Prediction speed is reported separately and is not the primary selection criterion.

## 6. Project Results

| Metric | Model A – DistilBERT | Model B – BERT |
|---|---:|---:|
| Accuracy | 0.8950 | 0.9250 |
| Precision | 0.8624 | 0.9126 |
| Recall | 0.9400 | 0.9400 |
| F1-score | 0.8995 | 0.9261 |
| Total prediction time (seconds) | 4.6217 | 9.1198 |
| Average latency (seconds/sample) | 0.0231 | 0.0456 |

### Final Decision

**Winner: Model B – BERT**

F1-score improvement: **0.0266**

Model B exceeds the required minimum improvement of 0.01. Therefore, Model B is selected as the Champion.

Although Model B has a higher prediction time, it achieves a better F1-score, which is the primary selection criterion.

## 7. MLflow Tracking and Model Registry

MLflow is used to:

- Track Model A and Model B evaluation runs.
- Log evaluation metrics and model parameters.
- Record the comparison and selection decision.
- Register both models under the name `Sentiment_Classifier`.
- Assign model aliases.

| Alias | Registered Version | Model |
|---|---:|---|
| champion | 1 | Model B – BERT |
| challenger | 2 | Model A – DistilBERT |

## 8. Prediction on New Data

The script `predict_from_mlflow.py` loads the model using the MLflow Champion alias:

`models:/Sentiment_Classifier@champion`

It predicts sentiment for 10 new sentences and saves the results to:

`outputs/new_data_predictions.csv`

The output contains the sentence, predicted sentiment, and confidence score.

## 9. Project Structure

```text
Champion_Challenger_Project/
│
├── data/
│   ├── SST-2/
│   └── evaluation_data.csv
│
├── outputs/
│   └── new_data_predictions.csv
│
├── screenshots/
│
├── config.py
├── hf_wrapper.py
├── evaluate.py
├── register_champion.py
├── predict_from_mlflow.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 10. How to Run the Project

### Step 1: Clone the repository

```bash
git clone <your-github-repository-url>
cd Champion_Challenger_Project
```

### Step 2: Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run model evaluation

```bash
python evaluate.py
```

### Step 5: Register models in MLflow

```bash
python register_champion.py
```

### Step 6: Start the MLflow UI

```bash
python -m mlflow ui --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` in your browser to view experiment tracking and registered models.

### Step 7: Predict on new sentences

```bash
python predict_from_mlflow.py
```

## 11. Limitations

- The evaluation uses 200 samples from the SST-2 validation dataset.
- Both models are pretrained models; no additional fine-tuning is performed.
- Prediction time may vary depending on hardware and system load.
- The models are designed for sentiment classification and may not perform equally well on every type of text.

## 12. AI Assistance Disclosure

AI assistance was used to help understand concepts, develop parts of the implementation, troubleshoot errors, and prepare project documentation. The project was executed and tested in the local Python environment.