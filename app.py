import numpy as np
import pandas as pd
import sklearn
import pickle
from flask import Flask, render_template, request

app = Flask(__name__,
            static_url_path='', 
            static_folder='web/static',
            template_folder='web/templates')

modelCost = pickle.load(open("modelCost.pkl", "rb"), encoding='bytes')
modelCase = pickle.load(open("modelCase.pkl", "rb"), encoding='bytes')

@app.route("/")
def home():
	return render_template('index.html')
	
@app.route("/predict/cost", methods=["POST"])
def predictCost():
    input_cost_prediction = [int(x) for x in request.form.values()]
    # convert input to numpy array
    arr_cost_prediction = np.array([input_cost_prediction])
    # convert numpy array to pandas DataFrame
    N = pd.DataFrame(arr_cost_prediction, columns = ['kddati2','peserta','case','rs'])
    N.index.name = 'row_id'
    predict = modelCost.predict(N)
    return render_template("index.html", reqCost = request.form, predict_cost = "{}".format(round(predict[0])))


@app.route("/predict/case", methods=["POST"])
def predictCase():
    input_case_prediction = [int(x) for x in request.form.values()]
    # convert input to numpy array
    arr_case_prediction = np.array([input_case_prediction])
    # convert numpy array to pandas DataFrame
    N = pd.DataFrame(arr_case_prediction, columns = ['kddati2','peserta','unit_cost','rs'])
    N.index.name = 'row_id'
    predict = modelCase.predict(N)
    return render_template("index.html", reqCase = request.form, predict_case = "{}".format(round(predict[0])))
	
if __name__ == "__main__":
	app.run(debug=True, use_reloader=True)