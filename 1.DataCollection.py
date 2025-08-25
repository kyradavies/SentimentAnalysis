import torch
import mediacloud.api as mc
from datetime import date
from datetime import datetime as dt
import pandas as pd
import calendar
import time
from sklearn.preprocessing import LabelEncoder
import os

# environment variable
api_key = os.environ.get("API_KEY_MEDIA_CLOUD")

UK_NATIONAL_COLLECTION = 34412476
UK_LOCAL=38381111
SOURCES_PER_PAGE = 100  # the number of sources retrieved per page
mc_search = mc.SearchApi(api_key)
# Now we can search for stories in the collection
# We will search for stories about Gen Z in the UK National Collection
# and limit the search to stories published between January 1, 2025 and January 5, 2025
query="""("Gen Z" OR "Generation Z" OR "young people" OR "youth" OR "teenagers") 
            AND ("attitudes" OR "opinions" OR "mindset" OR "values" OR "beliefs" OR "perceptions" 
            OR "lifestyle" OR "work ethic" OR "social views")"""
start_dates = [date(year, month, 1) for year in [i for i in range(2019,2025)] for month in range(1, 13)]
end_dates = [date(year, month,  calendar.monthrange(year, month)[1]) for year in [i for i in range(2019,2025)]  for month in range(1, 13)]

def fetch_data(start, end):
    all_stories = []
    pagination_token = None
    more_stories = True
    print(start,end)
    try:
        page, pagination_token = mc_search.story_list(
            query=query,
            start_date=start,
            end_date=end,
            collection_ids=[UK_NATIONAL_COLLECTION])
        all_stories += page
        more_stories = pagination_token is not None
        print(all_stories)
        df_temp = pd.DataFrame(all_stories)
        # Optional: rate limit
        time.sleep(1)  # pause between requests to avoid 403
        return df_temp
    except RuntimeError as e:
        if "403" in str(e):
            print("❌ 403 Forbidden API access denied or rate limited.")
        else:
            raise e  # for other unexpected errors

df=pd.DataFrame()
for start,end in zip(start_dates,end_dates):
    df_temp=fetch_data(start, end)
    if df_temp.empty:
        print(f"No data found for {start} to {end}")
        continue
    print(f"Fetched {len(df_temp)} stories from {start} to {end}")
    print(df_temp['indexed_date'].head())
    df=pd.concat([df,df_temp], ignore_index=True)
    time.sleep(61)   
print(df.columns)
df['indexed_date']= pd.to_datetime(df['indexed_date'])
print(df['publish_date'].min(), df['publish_date'].max())
#df['indexed_date']=df['indexed_date'].dt.strftime('%Y-%m')
df.to_csv('data_collection_genz2.csv', index=False)
#df.to_csv('sentiment_analysis_results.csv', mode='a', header=False, index=False)
