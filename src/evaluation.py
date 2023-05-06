from src.find_best_model import find_best_model
from src.utils.common_utils import log, read_params, clean_prev_dirs_if_exis, create_dir, save_model, save_report
from src.pre_processing import standardization
from src.load_and_save import load_data



def evaluation(config_path):
    """
        It helps to get the best model after evaluation.\n
        :param config_path: params.yaml
        :return: None
    """
    try:
        config = read_params(config_path=config_path) # read the params.yaml file
        evaluation_logs = config['artifacts']['log_files']['evaluation_log_file']  # artifacts/Logs/evaluation_logs.txt
        log(file_object=evaluation_logs, log_message="model evaluation process is start") # logs the details

        best_model_dir = config['artifacts']['model']['best_model']['best_model_dir'] # artifacts/Best_Model
        best_model_path = config['artifacts']['model']['best_model']['best_model_path'] # artifacts/Best_Model/model.joblib
        score_path = config['artifacts']['model']['best_model']['scores'] # artifacts/Best_Model/score.json

        clean_prev_dirs_if_exis(dir_path=best_model_dir) # remove artifacts/Best_Model directory if it is already created
        create_dir(dirs=[best_model_dir]) # create artifacts/Best_Model directory

        # get the train, test data from Processed_Data directory:
        train_path = config['artifacts']['processed_data']['train_path']  # artifacts/Processed_Data/train.csv
        test_path = config['artifacts']['processed_data']['test_path']  # artifacts/Processed_Data/test.csv
        output_col = config['data_defination']['output_col'] # default

        train = load_data(raw_data_path=train_path, log_file=evaluation_logs) # load the train data
        y_train = train[output_col] # get y_train data
        x_train = train.drop([output_col], axis=1) # get x_train data
        x_train = standardization(x_data=x_train, log_file=evaluation_logs) # scale the train data

        test = load_data(raw_data_path=test_path, log_file=evaluation_logs)  # load the test data
        y_test = test[output_col] # get y_test data
        x_test = test.drop([output_col], axis=1) # get the x_test data
        x_test = standardization(x_data=x_test, log_file=evaluation_logs) # scale x_test data

        random_forest_model_path = config['artifacts']['model']['random_forest']['random_forest_model_path']  # artifacts/Model/RandomForest/random_forest_model.joblib
        xgboost_model_path = config['artifacts']['model']['xgboost']['xgboost_model_path']  # artifacts/Model/XGBoost/xgboost_model.joblib

        model, dct = find_best_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                     random_forest_model_path=random_forest_model_path,
                                     xgboost_model_path=xgboost_model_path, log_file=evaluation_logs) # find the best model & get the dict
        save_model(model_name=model, model_path=best_model_path) # save the best model in artifacts/Best_Model
        save_report(file_path=score_path, report=dct) # save the model performance report in artifacts/Best_Model

    except Exception as e:
        print(e)
        config = read_params(config_path=config_path)  # read the params.yaml file
        evaluation_logs = config['artifacts']['log_files']['evaluation_log_file']  # artifacts/Logs/evaluation_logs.txt
        log(file_object=evaluation_logs, log_message=f"Error will be: {e}") # logs the details
        raise e
