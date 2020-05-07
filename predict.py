################################################################################################
#                           Importing dependencies here                                        #
################################################################################################
from joblib import load
from preprocess import prep_data
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


#################################################################################################
#                          Prepping data for prediction                                         #
#################################################################################################

def predict_from_csv(path_to_csv):

    df = pd.read_csv(path_to_csv)

    X, y = prep_data(df)

    reg = load("reg.joblib")

    predictions = reg.predict(X)

    return predictions


################################################################################################
#                          Testing using "fish_holdout_demo.csv"                               #
################################################################################################

if __name__ == "__main__":

    predictions = predict_from_csv("fish_holdout_demo.csv")
    y_truth = pd.read_csv("fish_holdout_demo.csv")["Weight"].values

    ho_mse = mean_squared_error(y_truth, predictions)
    r2_score = r2_score(y_truth, predictions)

    print(y_truth)
    print(predictions)
    print(ho_mse)
    print(r2_score)


### WE WRITE THIS ###
    # from sklearn.metrics import mean_squared_error
    # ho_predictions = predict_from_csv("fish_holdout.csv")
    # ho_truth = pd.read_csv("fish_holdout.csv")["Weight"].values
    # ho_mse = mean_squared_error(ho_truth, ho_predictions)
    # print(ho_mse)
######

