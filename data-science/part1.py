import pandas as pd

df = pd.read_csv('dataset/train.csv')

print(df.info())
print(df.describe())
print(df.head())

print(df.Operational_Notes.unique())