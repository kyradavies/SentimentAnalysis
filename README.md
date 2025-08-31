Got it — thanks for sharing all three scripts. I’ll create a **README.md draft** that explains what your project does, how to set it up, and how to run each stage. It’ll be written in clear, semi-formal UK English and structured for someone who may want to reproduce or adapt your workflow.

---

# README

## Overview

This project collects UK news headlines mentioning *Gen Z*, fine-tunes a sentiment analysis model, and applies the model to classify the sentiment of collected articles. The workflow is divided into three main parts:

1. **Data Collection** – Uses the [Media Cloud API](https://mediacloud.org) to gather headlines about *Gen Z* from UK national news outlets.
2. **Model Training** – Fine-tunes a DistilBERT sentiment classifier using the [Sentiment140 dataset](https://www.kaggle.com/kazanova/sentiment140).
3. **Evaluation and Inference** – Applies the fine-tuned model to the collected headlines, validates performance against a subset of Sentiment140, and produces sentiment labels.

---

## Requirements

* Python 3.9+
* GPU recommended (for model fine-tuning)

### Key dependencies

* `torch`
* `transformers`
* `datasets`
* `evaluate`
* `scikit-learn`
* `pandas`
* `mediacloud.api`
* `plotly` (for optional visualisations)
* `kagglehub` (for downloading Sentiment140)

Install requirements:

```bash
pip install torch transformers datasets evaluate scikit-learn pandas mediacloud-api plotly kagglehub
```

---

## 1. Data Collection

The first script (`data_collection.py`) retrieves news stories from Media Cloud.

* It searches the **UK National Collection** for headlines mentioning *Gen Z* (and related terms).
* Stories are collected month by month, from **2019 to 2025**.
* The data is saved as a CSV file (`data_collection_genz2.csv`).

### Run

Set your Media Cloud API key in the environment:

```bash
export API_KEY_MEDIA_CLOUD=your_api_key_here
```

Then run:

```bash
python data_collection.py
```

Output:

* `data_collection_genz2.csv` containing story metadata and headlines.

---

## 2. Model Training

The second script (`train_model.py`) fine-tunes a **DistilBERT** model for binary sentiment classification (positive vs negative).

* Training data: [Sentiment140](https://www.kaggle.com/kazanova/sentiment140).
* Labels: `0 = negative`, `4 → 1 = positive`.
* Data is shuffled and split into training and test sets.
* The model is fine-tuned using Hugging Face `Trainer`.
* The fine-tuned model is saved in the directory `sentiment-finetuned/`.

### Run

```bash
python train_model.py
```

Output:

* Fine-tuned model stored in `sentiment-finetuned/`
* Example inference with:

```python
from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="sentiment-finetuned")
print(sentiment_pipeline("I really liked that!"))
```

---

## 3. Evaluation and Inference

The third script (`evaluate_infer.py`) applies the fine-tuned model to:

* Headlines collected in **step 1**.
* A validation sample from **Sentiment140** (1% subset).

It then:

* Runs inference on the text data.
* Evaluates performance using accuracy, classification report, and confusion matrix.
* Provides scores and predicted labels.

### Run

```bash
python evaluate_infer.py
```

Output:

* Accuracy and classification report printed in console.
* Optionally, predictions can be saved to CSV by uncommenting relevant lines.

---

## Project Structure

```
.
├── data_collection.py        # Collects headlines from Media Cloud
├── train_model.py            # Fine-tunes DistilBERT sentiment model
├── evaluate_infer.py         # Applies model and evaluates performance
├── data_collection_genz2.csv # Collected dataset (created after running script 1)
├── sentiment-finetuned/      # Fine-tuned model (created after running script 2)
└── README.md
```

---

## Notes

* **Neutral class** is not included in the current training. To expand, consider fine-tuning with a dataset that includes neutral sentiment.
* Media Cloud has **rate limits**. The script includes pauses (`time.sleep`) to avoid 403 errors.
* GPU is recommended for fine-tuning. CPU training will be much slower.

---

Would you like me to also add **example usage commands** for running inference directly on a new CSV of headlines (so you don’t have to tweak the code manually)?
