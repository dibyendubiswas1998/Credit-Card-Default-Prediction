from src.utils.common_utils import log
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier



def apply_random_forest_classifier(x_train, y_train, random_state, log_file):
    """
        It helps to train the model based on RandomForestClassifier based on best params.\n
        :param x_train: x_train
        :param y_train: y_train
        :param random_state: random_state
        :param log_file: log_file
        :return: model
    """
    try:
        file = log_file
        # initializing with different combination of parameters
        # param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
        #               "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}
        #
        # base_rand = RandomForestClassifier() # base model
        # grid = GridSearchCV(estimator=base_rand, param_grid=param_grid, cv=5, verbose=3) # apply GridSearchCV
        # grid.fit(x_train, y_train)
        #
        # # extracting the best parameters:--
        # criterion = grid.best_params_['criterion']
        # max_depth = grid.best_params_['max_depth']
        # max_features = grid.best_params_['max_features']
        # n_estimators = grid.best_params_['n_estimators']
        #
        # random_forest = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
        #                                       max_depth=max_depth, max_features=max_features) # apply RandomForestClassifier based on best params
        # log(file_object=file, log_message=f"applying RandomForestClassifier algorithm with {grid.best_params_}") #logs the details

        random_forest = RandomForestClassifier(n_estimators=100, criterion='gini',
                                               max_depth=10, max_features='auto', random_state=random_state) # apply random forest
        random_forest.fit(x_train, y_train) # train the model based on train data
        log(file_object=file, log_message="apply the RandomForestClassifier() algorithm") # logs the details
        return random_forest # return the RandomForestClassifier model.

    except Exception as e:
        print(e)
        file = log_file
        log(file_object=file, log_message=f"Error will be: {e}")  # logs the error if occurs
        raise e


def apply_xgboost_classifier(x_train, y_train, log_file):
    """
        It helps to train the model based on XGBClassifier based on best params.\n
        :param x_train: x_train
        :param y_train: y_train
        :param log_file: log_file
        :return: model
    """
    try:
        file = log_file
        # param_grid_xgboost = {
        #     'learning_rate': [0.5, 0.1, 0.01, 0.001],
        #     'max_depth': [3, 5, 10, 20],
        #     'n_estimators': [10, 50, 100, 200]
        # }
        #
        # base_xgboost = XGBClassifier(objective='binary:logistic') # base model
        # grid = GridSearchCV(base_xgboost, param_grid_xgboost, verbose=3, cv=5) # apply GridSearchCV algorithm
        # grid.fit(x_train, y_train) # fit the data
        #
        # # extracting the best parameters:----
        # learning_rate = grid.best_params_['learning_rate']
        # max_depth = grid.best_params_['max_depth']
        # n_estimators = grid.best_params_['n_estimators']
        #
        # xgboost_model = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth,
        #                               n_estimators=n_estimators) # apply XGBClassifier algorithm based on best params.
        # xgboost_model.fit(x_train, y_train) # train the model.
        # log(file_object=file, log_message=f"applying XGBClassifier algorithm with {grid.best_params_}")  # logs the details

        xgboost_model = XGBClassifier(learning_rate=0.01, max_depth=10, n_estimators=100)  # apply XGBClassifier algorithm
        xgboost_model.fit(x_train, y_train) # train the model.
        log(file_object=file, log_message="apply the XGBClassifier() algorithm")  # logs the details
        return xgboost_model # return the model

    except Exception as e:
        print(e)
        file = log_file
        log(file_object=file, log_message=f"Error will be: {e}")  # logs the error if occurs
        raise e
