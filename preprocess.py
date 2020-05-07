###########################################################################################################################################################
# 														function to handle outliers (applied ONLY to training data)										  #
###########################################################################################################################################################


def handle_outlier(df):

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
        print(df1_outliers)  # <--- this printed row # 13

    # dropping the outlier at row # 13
    df = df.drop([13])

    return df


###########################################################################################################################################################
# 																 function to prep data for prediction													  #
###########################################################################################################################################################


def prep_data(df):

    # The 3 length variables are highly collinear. Length3 shows highest correlation with target variable and Length3 values are the highest for each row, it is probably the Total Length of the fish.
    # So keeping Length3 and dropping the other two length variables from the dataset.
    df = df.drop(["Length1", "Length2"], axis=1)

    # Dividing the data into features and target variable
    X = df.drop("Weight", axis=1)
    y = df["Weight"]

    return X, y
