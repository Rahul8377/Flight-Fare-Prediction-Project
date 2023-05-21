import pickle
import sys, os
from flight.exception import FlightException
from flight.logger import logging
from flight.credentials import *
from typing import *
import sys, os
import pandas as pd
from datetime import datetime

def predict_fare(df: Union[pd.DataFrame, List])->float:
    """
    This function will Predict the Flight Fair

    df: Pandas DataFrame
    ======================================
    returns Fair
    """
    try:
        logging.info(f"{'>>'*20} Predicting Flight Fare {'<<'*20}")

        # Load Model
        predict_fare = gen_model.predict(df)

        logging.info(f"Predicted Flight Fare={predict_fare[0]}")

        return float(round(predict_fare[0],2))

    except Exception as e:
        logging.debug(str(e))
        raise FlightException(error_message=e, error_detail=sys)


def transform_and_predict(data: List):
    """
    This function transform the collected information provided by the user into a pandas.DataFrame
    which contains all necessary features required by the Model for Prediction

    data: list of input data
    ==============================================================================================
    returns Pandas DataFrame
    """

    def source_encode(source:str, airline:str= None)->List:
        try:
            logging.info(f"Source Location: {source}")
            logging.info(f"Applying OneHotEncode on Source {source}")

            if airline is None:
                source_list = columns[8:13]
                logging.info(f"Before Encoding Source List: {source_list}")

                for i in range(len(source_list)):

                    if source in source_list[i]:
                        source_list[i] = 1
                    else:
                        source_list[i] = 0

                logging.info(f"After Encoding Source List: {source_list}")

            else:
                model = airline_model_dict[airline]
                source_list = []

                for feature_name in model.feature_names_in_:

                    if "Source" in feature_name:
                        if source == feature_name.split("_")[1]:
                            source_list.append(1)
                        else:
                            source_list.append(0)

            return source_list

        except Exception as e:
            logging.debug(str(e))
            raise FlightException(error_message=e, error_detail=sys)


    def destination_encode(destination:str, airline:str=None)->List:
        try:
            logging.info(f"Destination Location: {destination}")
            logging.info(f"Applying OneHotEncode on Destination {destination}")

            if airline is None:
                destination_list = columns[13:19]
                logging.info(f"Before Encoding Destination List: {destination_list}")

                for i in range(len(destination_list)):

                    if destination in destination_list[i]:
                        destination_list[i] = 1
                    else:
                        destination_list[i] = 0

                logging.info(f"After Encoding Destination List: {destination_list}")

            else:
                model = airline_model_dict[airline]
                destination_list = []

                for feature_name in model.feature_names_in_:

                    if "Destination" in feature_name:
                        if destination == feature_name.split("_")[1]:
                            destination_list.append(1)
                        else:
                            destination_list.append(0)

            return destination_list
        
        except Exception as e:
            logging.debug(str(e))
            raise FlightException(error_message=e, error_detail=sys)


    def airline_encode(airline:str)->List:
        try:
            logging.info(f"Airline: {airline}")
            logging.info(f"Applying OneHotEncode on Airline {airline}")

            airline_list = columns[19:]
            logging.info(f"Before Encoding Airline List: {airline_list}")

            for i in range(len(airline_list)):

                if airline in airline_list[i]:
                    airline_list[i] = 1
                else:
                    airline_list[i] = 0

            logging.info(f"BeforAftere Encoding Airline List: {airline_list}")

            return airline_list

        except Exception as e:
            logging.debug(str(e))
            raise FlightException(error_message=e, error_detail=sys)


    try:
        logging.info(f"{'>>'*20} Transforming Collecte Data {'<<'*20}")

        # data_trf = ['Duration', 'Total_Stops', 'Day', 'Month', 'Dep_hr', 'Dep_min', 'Arrival_hr', 'Arrival_min']
        # data = ['2023-01-30T12:34', '2023-01-30T03:05', 'Banglore', 'Delhi', '0', 'IndiGo']

        # DateTime Type Casting
        logging.info("STEP-1: Transforming Date And Time")
        dep_time = datetime.strptime(data[0], "%Y-%m-%dT%H:%M")
        ariv_time = datetime.strptime(data[1], "%Y-%m-%dT%H:%M")

        # Calculating Duration of Flight
        duration = int((ariv_time.timestamp() - dep_time.timestamp()) / 60)
        logging.info(f"duration of Flight: {duration}")

        # Adding First 8 features to data transformation list
        data_trf = [duration, int(data[4]), dep_time.day, dep_time.month, dep_time.hour, dep_time.minute, ariv_time.hour, ariv_time.minute]

        if data[-1] != "Any":

            logging.info("STEP-2: OneHotEncode Source, Destination, Airline")
            # OneHotEncode Source, Destination, Airline
            data_trf = data_trf + source_encode(data[2]) + destination_encode(data[3]) + airline_encode(data[-1])
            logging.info(f"Transformed Data: {data_trf}")

            # Converting the transformed data list into DataFrame with all Features Names as Column Names
            logging.info("Create input DataFrame for Model with Transformed Data")
            model_input = pd.DataFrame(columns=columns)
            model_input.loc[len(model_input.index)] = data_trf
            logging.info(f"Input DataFrame Shape: {model_input.shape}")
            result = predict_fare(df=model_input)
            logging.info(f"Predicted Fare When Source, Destination, Airline available: {result}")

        else:

            available_airline = avail_source_airlines[data[2]]
            logging.info(f"Available Airlines From Source: {data[2]} are:{available_airline}")

            result = {}

            for airline in available_airline:
                logging.info(f"Computing Fare for airline {airline}")
                data_encoded = data_trf + source_encode(source=data[2],airline=airline) + destination_encode(destination=data[3],airline=airline)
                logging.info(f"Transformed Data: {data_encoded}")

                model = airline_model_dict[airline]
                logging.info(f"Required features : {model.feature_names_in_}")

                # Converting the transformed data list into DataFrame with all Features Names as Column Names
                logging.info("Create input DataFrame for Model with Encoded Data")
                model_input = pd.DataFrame(columns=model.feature_names_in_)
                model_input.loc[len(model_input.index)] = data_encoded
                logging.info(f"Input DataFrame Shape: {model_input.shape}")
                fare = model.predict(model_input)
                logging.info(f"Predicted fare = {fare}")
                result[airline] = round(fare[0],2)
            
            logging.info(f"Predicted Fare When Only Source and Destination are available: {result}")

        logging.info(f"Final Predicted Fare={result}")
        return result

    except Exception as e:
            logging.debug(str(e))
            raise FlightException(error_message=e, error_detail=sys)