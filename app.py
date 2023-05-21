from flight.credentials import *
from flight.fair_calculate import predict_fare, transform_and_predict
from flight.exception import FlightException
from flight.logger import logging
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd


app = Flask(__name__)

@app.route('/')
def home():
    logging.info("Rendering Home Page")
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    logging.info("Collecting Data From api")
    data = request.json['data']
    logging.info("Converting collected data into 2-D array")
    new_data = np.array(list(data.values())).reshape(1,-1)
    logging.info("Predicting Flight Fare")
    prediction = predict_fare(new_data)
    logging.info("Showing Predicted Fare")
    logging.info(prediction[0])
    return jsonify(prediction[0])


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        logging.info(f"Collecting Data From Webpage")
        data = [x for x in request.form.values()]

        logging.info(f"Collected Info:{data}")

        result = transform_and_predict(data)

        if type(result) is dict:
            entry = {"Airline": result.keys(), "Fare (INR)":result.values()}
            data = pd.DataFrame(data=entry)
            data.sort_values(by="Fare (INR)", ascending=False, inplace=True)
            prediction_text = f"Predicted Flight Fare for available Airlines"

        elif type(result) is float:
            prediction_text = f"Predicted Flight Fare is INR {result}"
            data=pd.DataFrame()

        else:
            prediction_text="No Flights are available."
            data=pd.DataFrame()

        logging.info(f"******************** END *************************")

        return render_template("home.html", Prediction_Text=prediction_text, data=data.to_html(col_space=200, index=False, justify="left", border=0))

    else:

        return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)





