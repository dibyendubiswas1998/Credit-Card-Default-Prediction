from src.utils.common_utils import read_params, read_file
from src.training import training
from src.evaluation import evaluation
from src.prediction import prediction
from flask import Flask, render_template, request, send_file
import json

app =Flask(__name__)

@app.route("/")
def Home():
    return render_template('index.html')

@app.route("/train_evaluation", methods=['POST', 'GET'])
def Train():
    if request.method == 'POST':
        train_data_path = request.form['train']
        training(config_path="params.yaml", data_path=train_data_path) # training pipeline
        dct = evaluation(config_path="params.yaml")

        # get the best param.
        config = read_params("params.yaml")
        report_file = config['artifacts']['model']['best_model']['scores']  # artifacts/Model/Best_Model/score.json
        report_file = read_file(report_file)
        dct = json.loads(report_file)  # loads as dictionary format
        model_name = dct['Model Name']  # get the model name
        score = dct['test data'].get("accuracy score")  # get accuracy score

        status = f"Train Successfully with {model_name} & score {round(score,2)*100}%"
        return render_template("index.html", status1=status)


@app.route("/prediction", methods=['POST', 'GET'])
def Prediction():
    if request.method == 'POST':
        prediction_data_path = request.form['prediction']
        prediction(config_path="params.yaml", data_path=prediction_data_path) # for prediction pipeline
        config =  read_params("params.yaml")
        file_path = config['artifacts']['prediction']['prediction_file'] # artifacts/Prediction/predict.csv
        # file = send_file(file_path, as_attachment=True)
        return render_template('index.html', status2="Prediction is completed")

@app.route("/download")
def Download():
    config = read_params("params.yaml")
    file_path = config['artifacts']['prediction']['prediction_file']  # artifacts/Prediction/predict.csv
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)




