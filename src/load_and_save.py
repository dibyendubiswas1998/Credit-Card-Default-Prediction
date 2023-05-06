import pandas as pd
from src.utils.common_utils import log, save_raw_local_df

def load_data(raw_data_path, log_file):
    """
        get the data from Raw Data folder & store to artifacts directory:\n
        :param raw_data_path: raw_data_path
        :param directory_path: folder_name
        :param new_data_path: new_data_path
        :return: data
    """
    try:
        data = pd.read_csv(raw_data_path, sep=',') # read the data
        file = log_file

        # # Convert the column header in lower case:
        # data.columns = data.columns.str.lower()
        # data.columns = data.columns.str.replace(' ', '')
        #
        # # Convert string values in lowe case:
        # data = data.applymap(lambda s: s.lower() if type(s) == str else s)
        # data = data.applymap(lambda s: s.strip() if type(s) == str else s)

        log(file_object=file, log_message=f"read the data, shape: {data.shape}")  # logs the details
        return data


    except Exception as e:
        print(e)
        file = log_file
        log(file_object=file, log_message=f"Error will be: {e}")  # logs the error if occurs
        raise e

def save_data(data, new_data_path, log_file):
    """
        It helps to save the data.
        :param data: data
        :param new_data_path: new_data_path
        :return: save data
    """
    try:
        file = log_file
        data = data
        save_raw_local_df(data=data, data_path=new_data_path)
        log(file_object=file, log_message=f"store data in : {new_data_path}")  # logs the details
        return data  # return data

    except Exception as e:
        print(e)
        file = log_file
        log(file_object=file, log_message=f"Error will be: {e}")  # logs the error if occurs
        raise e

if __name__ == "__main__":
    pass