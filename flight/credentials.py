import pickle
import sys, os
from flight.exception import FlightException
from flight.logger import logging

try:
    logging.info(f"Loading All The Latest Models...")
    list_dir = os.listdir("saved_models")
    latest_dir_num = max(list_dir)
    latest_models_dir = os.path.join("saved_models",f"{latest_dir_num}")

    gen_model = pickle.load(open(os.path.join(latest_models_dir,"gen_model.pkl"),'rb'))
    JetAirways_model = pickle.load(open(os.path.join(latest_models_dir,"JetAirways_model.pkl"),'rb'))
    Indigo_model = pickle.load(open(os.path.join(latest_models_dir,"Indigo_model.pkl"),'rb'))
    AirIndia_model = pickle.load(open(os.path.join(latest_models_dir,"AirIndia_model.pkl"),'rb'))
    MultipleCarriers_model = pickle.load(open(os.path.join(latest_models_dir,"MultipleCarriers_model.pkl"),'rb'))
    SpiceJect_model = pickle.load(open(os.path.join(latest_models_dir,"SpiceJet_model.pkl"),'rb'))
    Vistara_model = pickle.load(open(os.path.join(latest_models_dir,"Vistara_model.pkl"),'rb'))
    AirAsia_model = pickle.load(open(os.path.join(latest_models_dir,"AirAsia_model.pkl"),'rb'))
    GoAir_model = pickle.load(open(os.path.join(latest_models_dir,"GoAir_model.pkl"),'rb'))

except Exception as e:
    logging.debug(str(e))
    raise FlightException(error_message=e, error_detail=sys)


columns = ['Duration', 'Total_Stops', 'Day',
       'Month', 'Dep_hr', 'Dep_min', 'Arrival_hr', 'Arrival_min',
       'Source_Banglore', 'Source_Chennai', 'Source_Delhi', 'Source_Kolkata',
       'Source_Mumbai', 'Destination_Banglore', 'Destination_Cochin',
       'Destination_Delhi', 'Destination_Hyderabad', 'Destination_Kolkata',
       'Destination_New Delhi', 'Airline_Air Asia', 'Airline_Air India',
       'Airline_GoAir', 'Airline_IndiGo', 'Airline_Jet Airways',
       'Airline_Jet Airways Business', 'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy']

airline_model_dict = {
    "Jet Airways":JetAirways_model,
    "IndiGo":Indigo_model,
    "Air India":AirIndia_model,
    "Multiple carriers":MultipleCarriers_model,
    "SpiceJet":SpiceJect_model,
    "Vistara":Vistara_model,
    "Air Asia":AirAsia_model,
    "GoAir": GoAir_model
}

avail_source_airlines = {
    'Banglore':['Jet Airways','IndiGo','Air India','Vistara','SpiceJet','GoAir','Air Asia'],
    'Kolkata':['Jet Airways','Air India','IndiGo','SpiceJet','Vistara','Air Asia','GoAir'],
    'Delhi':['Jet Airways','Multiple carriers','Air India','IndiGo','SpiceJet','Air Asia','GoAir','Vistara'],
    'Chennai':['IndiGo','SpiceJet','Vistara','Air India'],
    'Mumbai':['Jet Airways','IndiGo','Air India','SpiceJet','Vistara']
}
