from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
import mediacloud.api as mc
from datetime import date
from datetime import datetime as dt
import pandas as pd
import calendar
import time
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import os

df=pd.read_csv("data_collection_genz2.csv")
#df=df[:200]
def model(data):
    model_path = "sentiment-finetuned"
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    print(model.config.id2label)

    # Create sentiment pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    score=sentiment_pipeline(data)
    return score


def fine_tune_model(data):
    # Load the pre-trained model and tokenizer
    model_name = "sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # Create a sentiment analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    score=sentiment_pipeline(data)
    return score

df = df[df["title"].apply(lambda x: isinstance(x, str))]
data = df['title'].to_list()
#score= fine_tune_model(data)
score= model(data)

df['sentiment']=score
df["sentiment_label"]=df["sentiment"].apply(lambda x: int(x["label"].split("_")[1]))
df["sentiment_score"] = df['sentiment'].apply(lambda x: x['score'])
#df["sentiment_label"] = df[score][0]['label']
#df["sentiment_score"] = df[score][0]['score'] 
#['Irrelevant' 'Negative' 'Neutral' 'Positive']
df.loc[df['sentiment_label']==0,"label_name"]='Irrelevant'
df.loc[df['sentiment_label']==1,"label_name"]='Negative'
df.loc[df['sentiment_label']==2,"label_name"]='Neutral'
df.loc[df['sentiment_label']==3,"label_name"]='Positive'

print(df.head())
df.to_csv('sentiment_analysis_finalV2.csv', index=False)