import pandas as pd
from src.utils.common_utils import log
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def remove_duplicate(data, log_file):
    """
        It helps to remove the duplicated values.\n
        :param data: data
        :param log_file: log_file
        :return: data
    """
    try:
        file = log_file
        data.drop_duplicates(keep=False, inplace=True)
        log(file_object=file, log_message=f"remove the duplicates from data")
        return data # return data.

    except Exception as e:
        print(e)
        file = log_file
        log(file_object=file, log_message=f"Error will be: {e}")  # logs the details
        raise e


def handle_outliers(data, col, log_file):
    """
        It helps to remove the outliers using IQR method.\n
        :param data: data
        :param col: col
        :param log_file: log_file
        :return: data
    """
    try:
        data = data
        col = col
        file = log_file
        log(file_object=file, log_message=f"Handle the Outliers using IQR method, col: {col}")  # logs the details

        q1 = data[col].quantile(0.25) # 25-percentile
        q3 = data[col].quantile(0.75) # 75-percentile
        IQR = q3 - q1 # inter quantile range
        lower = q1 - 1.5 * IQR # lower limit
        upper = q3 + 1.5 * IQR # upper limit

        data.loc[data[col] >= upper, col] = upper
        data.loc[data[col] <= lower, col] = lower
        return data # return data after removing outliers

    except Exception as e:
        print(e)
        file = log_file
        log(file_object=file, log_message=f"Error will be: {e}")  # logs the details
        raise e


def over_sampling_smote(data, ycol, random_state, log_file):
    """
        It helps to balance the data using SMOTE technique.\n
        :param data: data
        :param ycol: ycol
        :param random_state: random_state
        :param log_file: log_file
        :return: balance data
    """
    try:
        file = log_file
        data = data
        ycol = ycol
        log(file_object=file, log_message=f"applying SMOTE technique to balance the data ----> oversampled")
        smote = SMOTE(random_state=random_state)  # by default neighbors = 5 (minority points)
        X = data.drop(axis=1, columns=ycol)
        x_cols = X.columns # get the all independent columns
        Y = data[ycol]

        X, Y = smote.fit_resample(X, Y)
        over_sample_data = pd.DataFrame(X, columns=x_cols)
        over_sample_data[ycol] = Y
        return over_sample_data # return data

    except Exception as e:
        print(e)
        file = log_file
        log(file_object=file, log_message=f"Error will be: {e}")  # logs the details
        raise e


def standardization(x_data, log_file):
    """
        It helps to apply standardization.\n
        :param x_data: x_train or x_test data
        :return: x_data
    """
    try:
        file = log_file
        log(file_object=file, log_message=f"Apply the standardization")  # logs the details
        scaler = StandardScaler() # apply standardization
        x_data = scaler.fit_transform(x_data)  # scaled the data using StandardScaler() method
        return x_data # return standardized data

    except Exception as e:
        print(e)
        file = log_file
        log(file_object=file, log_message=f"Error will be: {e}")  # logs the details
        raise e

