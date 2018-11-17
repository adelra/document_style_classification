import pandas as pd

df = pd.read_csv('news_train.csv')
labels = df.values[:,0]
features = df.values[:,1]