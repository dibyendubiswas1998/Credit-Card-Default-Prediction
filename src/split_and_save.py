from src.utils.common_utils import  log, clean_prev_dirs_if_exis, create_dir, save_raw_local_df
from sklearn.model_selection import train_test_split


def split_and_save_data(data, log_file, split_ratio, random_state, directory_path, train_data_path, test_data_path):
    """
    It helps to split data.\n
    :param data: data
    :param log_file: log_file.txt
    :param split_ratio: split_ratio
    :param random_state: random_state
    :param directory_path: directory_path
    :param train_data_path: train_data_path
    :param test_data_path: test_data_path
    :return: train & evaluation
    """
    try:
        data = data
        file = log_file

        train, test = train_test_split(data, test_size=split_ratio, random_state=random_state) # split data in train & test set
        log(file_object=file, log_message=f"split ghe data in tran.csv: {train.shape} & test.csv: {test.shape}") # logs the details

        clean_prev_dirs_if_exis(dir_path=directory_path)  # cleaned directory if exists
        create_dir(dirs=[directory_path])  # create directory.
        log(file_object=file,
            log_message=f"create directory for storing the data: {directory_path}")  # logs the details

        for data, data_path in (train, train_data_path), (test, test_data_path):
            save_raw_local_df(data, data_path)
            log(file_object=file, log_message=f"store data in : {data_path}")  # logs the details
        return train, test  # return train, test dataset

    except Exception as e:
        print(e)
        file = log_file
        log(file_object=file, log_message=f"Error will be: {e}")  # logs the error if occurs
        raise e


if __name__ == '__main__':
    pass
