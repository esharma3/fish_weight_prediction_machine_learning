import numpy as np
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder

def prep_data(df):

	numerical = ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]

	# finding the outliers
	for col in numerical:
		df1 = df[col]
		df1_Q1 = df1.quantile(0.25)
		df1_Q3 = df1.quantile(0.75)
		df1_IQR = df1_Q3 - df1_Q1
		df1_lowerend = df1_Q1 - (1.5 * df1_IQR)
		df1_upperend = df1_Q3 + (1.5 * df1_IQR)

		df1_outliers = df1[(df1 < df1_lowerend) | (df1 > df1_upperend)]
		# print(df1_outliers)

    # dropping the outlier at row # 13
	# df = df.drop([13])

	# applying log conversion to normalizes the distribution of features and the target variable to some extent
	df["Weight"] = np.log(df["Weight"])
	df["Length1"] = np.log(df["Length1"])
	df["Length2"] = np.log(df["Length2"])
	df["Length3"] = np.log(df["Length3"])
	df["Height"] = np.log(df["Height"])
	df["Width"] = np.log(df["Width"])

	# since the 3 lenght variables are highly correlated and Length3 values are the highest for each row, it is probably the Total Length of the fish. 
	# So keeping Length3 and dropping the other two length variables from the dataset.
	df = df.drop(["Length1", "Length2"], axis=1)

	# dividing the data into features and target variable
	X = df.drop("Weight", axis=1).values
	y = df["Weight"].values

	# encoding the categorical feature - species
	ct = ColumnTransformer(
	    [("one_hot_encoder", OneHotEncoder(categories="auto"), [0])],
	    remainder="passthrough",
	)

	X = ct.fit_transform(X)

	return X, y