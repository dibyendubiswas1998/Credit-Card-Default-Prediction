from src.utils.common_utils import log, read_params, clean_prev_dirs_if_exis, create_dir
from src.load_and_save import load_data, save_data
from src.pre_processing import standardization, handle_outliers, remove_duplicate
import pandas as pd
import joblib

def prediction(config_path, data_path):
    """
        It helps to predict the based on new data through web app.\n
        :param config_path: config_path (params.yaml)
        :param data_path: data_path
        :return: None
    """
    try:
        config = read_params(config_path=config_path)  # read the params.yaml file
        prediction_logs = config['artifacts']['log_files']['prediction_file']  # artifacts/Logs/prediction_logs.txt
        log(file_object=prediction_logs, log_message=f"prediction process start") # logs the details

        prediction_dir = config['artifacts']['prediction']['prediction_dir'] # artifacts/Prediction directory
        prediction_file_path = config['artifacts']['prediction']['prediction_file'] # artifacts/Prediction/predict.csv

        clean_prev_dirs_if_exis(dir_path=prediction_dir) # remove artifacts/Prediction directory if it is already created
        create_dir(dirs=[prediction_dir]) # create artifacts/Prediction directory
        log(file_object=prediction_logs, log_message=f"create a directory for storing the data, path: {prediction_dir}") # logs the details

        data = load_data(raw_data_path=data_path, log_file=prediction_logs) # load the data
        cols = data.columns # get the columns

        # handle the duplicates:
        data = remove_duplicate(data=data, log_file=prediction_logs) # remove duplicates
        # handle the outliers:
        for col in cols:
            data = handle_outliers(data=data, col=col, log_file=prediction_logs)

        data_scale = standardization(x_data=data, log_file=prediction_logs) # scale the data
        model_path = config['artifacts']['model']['best_model']['best_model_path'] # artifacts/Best_Model/model.joblib
        model = joblib.load(model_path) # load the model
        log(file_object=prediction_logs, log_message=f"predict the output based on best model") # logs the details

        y_predict = model.predict(data_scale)
        new_data = pd.DataFrame(data, columns=cols)
        new_data['default'] = y_predict
        save_data(data=new_data, new_data_path=prediction_file_path, log_file=prediction_logs)
        log(file_object=prediction_logs, log_message=f"prediction is completed & result is store in {prediction_file_path}")

    except Exception as e:
        print(e)
        config = read_params(config_path=config_path)  # read the params.yaml file
        prediction_logs = config['artifacts']['log_files']['prediction_file']  # artifacts/Logs/prediction_logs.txt
        log(file_object=prediction_logs, log_message=f"error will be: {e}")  # logs the details
        raise e

