import pandas as pd
from sklearn.datasets import load_diabetes
import numpy as np

data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='disease_progression')

print("Feature Sample:")
print(X.head())

print("Target Sample:")
print(y.head())

df = pd.concat([X, y], axis=1)
df.to_csv("diabetes_dataset.csv", index=False)