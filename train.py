####################################################################################
# 						 importing dependencies here                               #
####################################################################################
import numpy as np
import pandas as pd

# for data prep
from preprocess import prep_data, handle_outlier

# for encoding and transformation
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import TransformedTargetRegressor

# for building pipeline
from sklearn.pipeline import make_pipeline

# model
from sklearn.linear_model import LinearRegression

# for saving the model
from joblib import dump


################################################################################
# 							     MAIN 										   #
################################################################################

df = pd.read_csv("fish_participant.csv")

df = handle_outlier(df)

X, y = prep_data(df)

# for encoding the categorical feature(species) and for transforming the numerical features
numerical = ["Length3", "Height", "Width"]
categorical = ["Species"]

preprocesser = ColumnTransformer(
    transformers=[
        ("one_hot_encoder", OneHotEncoder(), categorical),
        ("log_transformer", FunctionTransformer(np.log, validate=False), numerical,),
    ],
    remainder="passthrough",
)

# building the pipeline
lr_pipe = make_pipeline(
    preprocesser,
    TransformedTargetRegressor(
        regressor=LinearRegression(), func=np.log, inverse_func=np.exp
    ),
)

# fitting the data
lr_pipe.fit(X, y)

# saving the model
dump(lr_pipe, "reg.joblib")
