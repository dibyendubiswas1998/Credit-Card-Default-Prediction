from src.utils.common_utils import log, read_params, clean_prev_dirs_if_exis, create_dir, save_model
from src.load_and_save import load_data, save_data
from src.pre_processing import remove_duplicate, over_sampling_smote, handle_outliers, standardization
from src.split_and_save import split_and_save_data
from src.clustering import elbow_method, create_clustering
from src.model_creation import apply_random_forest_classifier, apply_xgboost_classifier


def training(config_path, data_path=None):
    """
        It helps to handle the data & create the model.\n
        :param config_path: params.yaml
        :param data_path: custom_data_path
        :return: None
    """
    try:
        config = read_params(config_path=config_path) # read the params.yaml file
        data_path = data_path # load the data path
        training_logs = config['artifacts']['log_files']['training_log_file'] # artifacts/Logs/training_logs.txt


        # Step 01: load & save the data:
        log(file_object=training_logs, log_message=f"load & save operation start") #logs the details
        data = load_data(raw_data_path=data_path, log_file=training_logs) # load the data

        new_data_dir = config['artifacts']['raw_data']['raw_data_dir'] # artifacts/Raw_Data
        new_raw_data_path = config['artifacts']['raw_data']['new_raw_data_path'] # artifacts/Raw_Data/new_data.csv

        clean_prev_dirs_if_exis(dir_path=new_data_dir) # remove artifacts/Raw_Data directory if it is already present.
        create_dir(dirs=[new_data_dir]) # create artifacts/Raw_Data directory

        save_data(data=data, new_data_path=new_raw_data_path, log_file=training_logs) # save the data in artifacts/Raw_Data directory
        log(file_object=training_logs, log_message="load & save process is completed successfully\n") # logs the details


        # Step 02: Pre-processed the data:
        log(file_object=training_logs, log_message=f"pre-processed operation is start") # logs the details
        new_raw_data_path = config['artifacts']['raw_data']['new_raw_data_path']  # artifacts/Raw_Data/new_data.csv
        data = load_data(raw_data_path=new_raw_data_path, log_file=training_logs) # load the data

        # remove duplicates:
        data = remove_duplicate(data=data, log_file=training_logs) # get the data after remove duplicates

        # to balance the data:
        output_col = config['data_defination']['output_col'] # default feature
        random_state = config['split']['random_state']  # random state
        data = over_sampling_smote(data=data, ycol=output_col, random_state=random_state, log_file=training_logs) # balance the data

        # handle outliers:
        for col in data.columns:
            data = handle_outliers(data=data, col=col, log_file=training_logs) # handle the outliers using IQR method.
        log(file_object=training_logs, log_message=f"Pre-processing operation is completed successfully\n") # logs the details


        # Step 03: Splitting Data:
        log(file_object=training_logs, log_message=f"data splitting operation is start") # logs the details
        processed_dir = config['artifacts']['processed_data']['processed_dir'] # artifacts/Processed_Data directory
        train_path = config['artifacts']['processed_data']['train_path'] # artifacts/Processed_Data/train.csv
        test_path = config['artifacts']['processed_data']['test_path'] # artifacts/Processed_Data/test.csv

        clean_prev_dirs_if_exis(dir_path=processed_dir) # remove artifacts/Processed_Data directory if it is already created.
        create_dir(dirs=[processed_dir]) # create artifacts/Processed_Data directory

        split_ratio = config['split']['split_ratio'] # split the data based on split ratio
        random_state = config['split']['random_state'] # split the data based on random state
        train, test = split_and_save_data(data=data, log_file=training_logs, split_ratio=split_ratio,
                                          random_state=random_state, directory_path=processed_dir,
                                          train_data_path=train_path, test_data_path=test_path) # save the data in train & test
        log(file_object=training_logs, log_message=f"splitting operation is completed successfully\n") # logs the details


        # Step 04: Model Training:
        log(file_object=training_logs, log_message=f"model training process is start based on train data") #logs the details
        output_col = config['data_defination']['output_col'] # default feature
        y_train = train[output_col] # get the independent features data
        x_train = train.drop([output_col], axis=1) # get the dependent features data
        x_train_scale = standardization(x_data=x_train, log_file=training_logs) # scale (standardization) the x_train data

        model_dir = config['artifacts']['model']['model_dir'] # artifacts/Model
        clean_prev_dirs_if_exis(dir_path=model_dir) # remove artifacts/Model directory if it is already created
        create_dir(dirs=[model_dir]) # create artifacts/Model directory
        log(file_object=training_logs, log_message=f"create directory for store the models, path: {model_dir}") # logs the details

        xgboost_dir = config['artifacts']['model']['xgboost']['xgboost_dir']  # artifacts/Model/XGBoost directory
        clean_prev_dirs_if_exis(dir_path=xgboost_dir) # remove artifacts/Model/XGBoost directory if it is already created
        create_dir(dirs=[xgboost_dir]) # create artifacts/Model/XGBoost directory
        log(file_object=training_logs, log_message=f"create directory for store the XGBoost model, path: {xgboost_dir}") # logs the details

        random_forest_dir = config['artifacts']['model']['random_forest']['random_forest_dir'] # artifacts/Model/RandomForest directory
        clean_prev_dirs_if_exis(dir_path=random_forest_dir) # remove artifacts/Model/RandomForest directory if it is already created
        create_dir(dirs=[random_forest_dir]) # create artifacts/Model/RandomForest directory
        log(file_object=training_logs, log_message=f"create directory for store the RandomForest model, path: {random_forest_dir}") # logs the details

        random_forest_model_path = config['artifacts']['model']['random_forest']['random_forest_model_path'] # artifacts/Model/RandomForest/random_forest_model.joblib
        xgboost_model_path = config['artifacts']['model']['xgboost']['xgboost_model_path']  # artifacts/Model/XGBoost/xgboost_model.joblib

        # create models:
        random_forest_model = apply_random_forest_classifier(x_train=x_train_scale, y_train=y_train, random_state=random_state,
                                                             log_file=training_logs) # create random forest model
        save_model(model_name=random_forest_model, model_path=random_forest_model_path) # save the random forest model in artifacts/Model/RandomForest directory
        log(file_object=training_logs, log_message=f"create random_forest model and save into {random_forest_model_path}") # logs the details

        xgboost_model = apply_xgboost_classifier(x_train=x_train_scale, y_train=y_train, log_file=training_logs) # create xgboost model
        save_model(model_name=xgboost_model, model_path=xgboost_model_path) # save the xgboost model in artifacts/Model/XGBoost/xgboost_model.joblib
        log(file_object=training_logs, log_message=f"create xgboost model and save into {xgboost_model_path}") # logs the details
        log(file_object=training_logs, log_message=f"model creation or model training process is completed successfully\n") # logs the details

    except Exception as e:
        print(e)
        config = read_params(config_path=config_path)  # read the params.yaml file
        training_logs = config['artifacts']['log_files']['training_log_file']  # artifacts/Logs/training_logs.txt
        log(file_object=training_logs, log_message=f"error will be: {e}") # logs the details
        raise e
