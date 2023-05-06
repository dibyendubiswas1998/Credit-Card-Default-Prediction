from src.utils.common_utils import log
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, confusion_matrix
import joblib

def find_best_model(x_train, y_train, x_test, y_test, random_forest_model_path, xgboost_model_path, log_file):
    """
        It helps to find the best model.\n
        :param x_train: x_train
        :param y_train: y_train
        :param x_test: x_test
        :param y_test: y_test
        :param random_forest_model_path: random_forest_model_path
        :param xgboost_model_path: xgboost_model_path
        :param log_file: log_file
        :return: best_model
    """
    try:
        file = log_file
        log(file_object=file, log_message=f"load the models XGBoost: {xgboost_model_path}, RandomForest: {random_forest_model_path}") # logs the details
        random_forest = joblib.load(random_forest_model_path) # load the random forest model
        xgboost_model = joblib.load(xgboost_model_path) # load the xgboost model

        # Comparing the models based on test data:
        y_predict_random_test = random_forest.predict(x_test) # predict the value based on test data using RandomForest algorithm
        y_predict_xgboost_test = xgboost_model.predict(x_test) # predict the value based on test data using XGBoost algorithm

        # XGBoost: if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
        if len(y_test.unique()) == 1:
            xgboost_score = accuracy_score(y_test, y_predict_xgboost_test) # get the accuracy score based on XGBoost
            log(file_object=file, log_message=f"accuracy score based on XGBoost: {str(xgboost_score)}") #logs the details
        else:
            xgboost_score = roc_auc_score(y_test, y_predict_xgboost_test)  # get the accuracy score based on XGBoost
            log(file_object=file, log_message=f"roc_auc_score based on XGBoost: {str(xgboost_score)}")  # logs the details

        # RandomForest: if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
        if len(y_test.unique()) == 1:
            random_forest_score = accuracy_score(y_test, y_predict_random_test)  # get the accuracy score based on XGBoost
            log(file_object=file,
                log_message=f"accuracy score based on XGBoost: {str(random_forest_score)}")  # logs the details
        else:
            random_forest_score = roc_auc_score(y_test, y_predict_random_test)  # get the accuracy score based on XGBoost
            log(file_object=file,
                log_message=f"roc_auc_score based on XGBoost: {str(random_forest_score)}")  # logs the details

        # compare best on score (test data):
        if random_forest_score > xgboost_score: # RandomForest
            y_predict_random_train = random_forest.predict(x_train) # predict x_train data based on train data
            acc_score_train = accuracy_score(y_train, y_predict_random_train) # get accuracy score based on train data
            recall_score_train = recall_score(y_train, y_predict_random_train) # get the recall score based on train data
            precision_score_train = precision_score(y_train, y_predict_random_train) # get the precision score based on train data
            confusion_matrix_train = confusion_matrix(y_train, y_predict_random_train)  # get the confusion matrix based on train data

            recall_score_test = recall_score(y_test, y_predict_random_test)  # get the recall score based on test data
            precision_score_test = precision_score(y_test, y_predict_random_test)  # get the precision score based on test data
            confusion_matrix_test = confusion_matrix(y_test, y_predict_random_test)  # get the confusion matrix based on test data
            # store the result in dictionary
            dct = {
                "Model Name": "RandomForestClassifier",
                "train data": {
                    'accuracy score': acc_score_train,
                    'recall score': recall_score_train,
                    'precision score': precision_score_train,
                    # 'confusion matrix': confusion_matrix_train
                },
                "test data": {
                    'accuracy score': random_forest_score,
                    'recall score': recall_score_test,
                    'precision score': precision_score_test,
                    # 'confusion matrix': confusion_matrix_test
                }
            }
            log(file_object=file, log_message=f"best model is RandomForestClassifier, with score: {dct}") # logs the details
            return random_forest, dct

        else: # XGBoost
            y_predict_xgboost_train = xgboost_model.predict(x_train)  # predict x_train data based on train data
            acc_score_train = accuracy_score(y_train, y_predict_xgboost_train)  # get accuracy score based on train data
            recall_score_train = recall_score(y_train, y_predict_xgboost_train)  # get the recall score based on train data
            precision_score_train = precision_score(y_train, y_predict_xgboost_train)  # get the precision score based on train data
            confusion_matrix_train = confusion_matrix(y_train, y_predict_xgboost_train) # get the confusion matrix based on train data

            recall_score_test = recall_score(y_test, y_predict_xgboost_test)  # get the recall score based on test data
            precision_score_test = precision_score(y_test, y_predict_xgboost_test)  # get the precision score based on test data
            confusion_matrix_test = confusion_matrix(y_test, y_predict_xgboost_test) # get the confusion matrix based on test data
            # store the result in dictionary
            dct = {
                "Model Name": "XGBoostClassifier",
                "train data": {
                    'accuracy score': acc_score_train, 'recall score': recall_score_train,
                    'precision score': precision_score_train,
                    # 'confusion matrix': confusion_matrix_train
                },
                "test data": {
                    'accuracy score': random_forest_score, 'recall score': recall_score_test,
                    'precision score': precision_score_test,
                    # 'confusion matrix': confusion_matrix_test
                }
            }
            log(file_object=file,
                log_message=f"best model is XGBoost, with score: {dct}")  # logs the details
            return xgboost_model, dct

    except Exception as e:
        print(e)
        file = log_file
        log(file_object=file, log_message=f"Error will be: {e}")  # logs the error if occurs
        raise e
