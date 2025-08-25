# Install dependencies
# pip install transformers datasets evaluate accelerate
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import evaluate
import pandas as pd
from sklearn.model_selection import train_test_split
import kagglehub
from datasets import Dataset
from datasets import DatasetDict

# Download latest version
path = kagglehub.dataset_download("jp797498e/twitter-entity-sentiment-analysis")

print("Path to dataset files:", path)

# Load your dataset (CSV with 'text' and 'label' columns)
#df=pd.read_csv("/Users/kyra/Library/Mobile Documents/com~apple~CloudDocs/Projects/Sentiment Analysis/sentiment_data/Sentiments/guardian_sentiment.csv")
df=pd.read_csv(f"{path}/twitter_training.csv")
df=df[['im getting on borderlands and i will murder you all ,','Positive']].rename(columns={ 'Positive': 'label','im getting on borderlands and i will murder you all ,': 'Headlines'})
# 4. Encode labels as integers
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])
train_dataset = Dataset.from_pandas(df)

#dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2)
dataset=pd.read_csv(f"{path}/twitter_validation.csv")
dataset=dataset[[ "I mentioned on Facebook that I was struggling for motivation to go for a run the other day, which has been translated by Tom’s great auntie as ‘Hayley can’t get out of bed’ and told to his grandma, who now thinks I’m a lazy, terrible person 🤣",'Irrelevant']].rename(columns={ 'Irrelevant': 'label',"I mentioned on Facebook that I was struggling for motivation to go for a run the other day, which has been translated by Tom’s great auntie as ‘Hayley can’t get out of bed’ and told to his grandma, who now thinks I’m a lazy, terrible person 🤣": 'Headlines'})
dataset["label"] = label_encoder.transform(dataset["label"])

test_dataset = Dataset.from_pandas(dataset)
print(dataset['Headlines'].str.len().max())



dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
print(dataset['train'][:10])
# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

#Dropping non-string entries
indices_to_remove = [i for i, text in enumerate(dataset["train"]["Headlines"]) if not isinstance(text, str)]
print(f"Removing {len(indices_to_remove)} non-string entries from training set.")
indices_to_remove_test = [i for i, text in enumerate(dataset["test"]["Headlines"]) if not isinstance(text, str)]
print(f"Removing {len(indices_to_remove_test)} non-string entries from test set.")  
dataset["train"] = dataset["train"].select([i for i in range(len(dataset["train"])) if i not in indices_to_remove])
dataset["test"] = dataset["test"].select([i for i in range(len(dataset["test"])) if i not in indices_to_remove_test])


# Tokenize data
def preprocess_function(examples):
    return tokenizer(examples["Headlines"], truncation=True, padding="max_length")

encoded_dataset = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Evaluation metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Save
trainer.save_model("sentiment-finetuned")
tokenizer.save_pretrained("sentiment-finetuned")

# Inference
from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="sentiment-finetuned", tokenizer="sentiment-finetuned")
print(sentiment_pipeline("I really liked that!"))
