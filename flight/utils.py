import pandas as pd
from flight.config import mongo_client
from flight.exception import FlightException
from flight.logger import logging
import os,sys
import yaml
import dill


def get_collection_as_dataframe(database_name:str, collection_name:str)->pd.DataFrame:
    """
    Description: This function return Collection as DataFrame
    Params:
    database_name: database name
    collection_name: collection name
    ===================================
    return Pandas DataFrame of a collection
    """
    try:
        logging.info(f"Reading data from database: {database_name} and collection: {collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"Found Columns: {df.columns}")
        if "_id" in df.columns:
            logging.info(f"Dropping column: _id")
            df = df.drop("_id",axis=1)
        logging.info(f"Row and Columns in df: {df.shape}")
        return df
    except Exception as e:
        logging.debug(str(e))
        raise FlightException(e, sys)


def write_yaml_file(file_path, data:dict):
    """
    This function create yaml file and write report in yaml file

    file_path: path of the file
    data: yaml file
    =================================================================
    this function does not return anything
    """
    try:
        file_dir = os.path.dirname(file_path)

        os.makedirs(file_dir, exist_ok=True)
        with open(file_path,'w') as file_writer:
            yaml.dump(data,file_writer)

    except Exception as e:
        logging.debug(str(e))
        raise FlightException(e, sys)

def save_object(file_path:str, obj:object):
    """
    This function will save the model

    file_path: Location or Path of Model where it will save
    obj: Model which have to save
    ================================
    return: none
    """
    try:
        logging.info("Entered the save_object method of Utils")
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
        logging.info("Exited the save_object method of Utils")

    except Exception as e:
        logging.debug(str(e))
        raise FlightException(e, sys)


def load_object(file_path:str)->object:
    """
    This function will load the model 

    file_path: Location or Path of Model where it is saved
    ================================
    return: it will return the model
    """
    try:
        if not os.path.exists(file_path):
            raise SensorException(f"The file: {file_path} is not exsist", sys)
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)

    except Exception  as e:
        logging.debug(str(e))
        raise SensorException(e, sys) from e

    
