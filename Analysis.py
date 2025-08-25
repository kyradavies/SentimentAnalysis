from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import mediacloud.api as mc
from datetime import date
from datetime import datetime as dt
import pandas as pd
import calendar
import time
import plotly.express as px

df=pd.read_csv("sentiment_analysis_finalV2.csv")
print(df['publish_date'].min(), df['publish_date'].max())
print(df[:20])
#df=pd.read_csv("sentiment_analysis_final.csv")
print(len(df))
df.drop_duplicates(subset=[i for i in df.columns if i !='publish_date'], inplace=True)
print(len(df))
print(df.info())
df['publish_date'] =df['publish_date'].astype(str).str[:10]
df['publish_date'] = pd.to_datetime(df['publish_date'])
#df['publish_date'] = pd.to_datetime(df['publish_date'])
df['publish_date'] = df['publish_date'].dt.to_period('Q')
df['publish_date'] = df['publish_date'].dt.to_timestamp()
df=df.loc[df['label_name']!='Irrelevant']
df_article=df.groupby(['publish_date', 'label_name'])['id'].nunique().reset_index(name='article_count')
df_article2=df.groupby(['publish_date'])['id'].nunique().reset_index(name='total_article_count')
df_article=df_article.merge(df_article2, on=['publish_date', ], how='left')
df_article["%"]=df_article['article_count']/df_article['total_article_count']*100
#df_article['rolling_avg'] = df_article['%'].rolling(window=6,min_periods=6).mean()
#df_article['publish_date'] = pd.to_datetime(df_article['publish_date'], format='%Y-%m')
print(df_article.info())

#fig = px.bar(df_article, x="publish_date", y="%", color="label_name", title="Article Sentiment Over Time",)
fig = px.scatter(df, x="publish_date", y="sentiment_score", color="label_name", symbol="label_name", title="Article Sentiment Over Time",)
fig.show()
#df.to_csv('sentiment_analysis_results.csv', index=False)
