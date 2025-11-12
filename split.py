import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

data = []
with open("News_Category_Dataset_v3.json", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

df = pd.DataFrame(data)
df = df[['headline', 'category', 'short_description']]

# take top n categories only 
#  ['COMEDY' 'PARENTING' 'SPORTS' 'ENTERTAINMENT' 'POLITICS' 'WELLNESS'
#  'BUSINESS' 'STYLE & BEAUTY' 'FOOD & DRINK' 'QUEER VOICES' 'HOME & LIVING'
#  'BLACK VOICES' 'TRAVEL' 'PARENTS' 'HEALTHY LIVING']

n = 15
top_categories = df['category'].value_counts().nlargest(n).index
df_small = df[df['category'].isin(top_categories)]

train_df, valid_df = train_test_split(df_small, test_size=0.2, random_state=42, stratify=df_small["category"])
valid_df, test_df = train_test_split(valid_df, test_size=0.5, random_state=42, stratify=valid_df["category"])


train_df.to_csv("train.csv", index=False, quoting = 1)
valid_df.to_csv("validation.csv", index=False, quoting = 1)
test_df.to_csv("test.csv", index=False, quoting = 1)


print("Saved train.csv and valid.csv in folder.")
print(f"Train size: {len(train_df)}, Validation size: {len(valid_df)}, Test size: {len(test_df)}") # Train size: 118497, Validation size: 14812, Test size: 14813