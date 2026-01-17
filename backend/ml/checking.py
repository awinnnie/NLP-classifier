import pandas as pd

df = pd.read_csv("Data/train.csv", quoting=1)

category_counts = df['category'].value_counts()
print(category_counts)

category_percent = df['category'].value_counts(normalize=True) * 100
print(category_percent)
