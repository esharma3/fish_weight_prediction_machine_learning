import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import dump
from preprocess import prep_data

from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline

df = pd.read_csv("fish_participant.csv")

print(df.shape)

X, y = prep_data(df)

# Linear Regression
lr = LinearRegression()
# pipe = make_pipeline(
#     QuantileTransformer(output_distribution="normal"), LinearRegression()
# )

lr.fit(X, y)

# saving the model
dump(lr, "reg.joblib")