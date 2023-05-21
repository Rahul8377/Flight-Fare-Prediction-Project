from flight.entity import config_entity
from flight.entity import artifact_entity
from flight.exception import FlightException
from flight.logger import logging
from flight import utils
from sklearn.preprocessing import LabelEncoder
import os,sys
import pandas as pd 
import numpy as np


class DataTransformation:

    def __init__(self, 
                    data_transformation_config:config_entity.DataTransformationConfig,
                    data_ingestion_artifact:artifact_entity.DataIngestionArtifact):

        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact

        except Exception as e:
            logging.debug(str(e))
            raise FlightException(error_message=e, error_detail=sys)

    
    # Function to transform date, time and duration features
    def transform_datetime_features(self, df:pd.DataFrame)->pd.DataFrame:
        """
        This function will transform date, time and duration features

        df: train.csv/test.csv
        ==============================================================
        returns Pandas Dataframe
        """
        try:
            # change data type of Date_of_Journey from object to datetime
            df.Date_of_Journey = pd.to_datetime(df.Date_of_Journey, dayfirst=True)

            # Creating Day and Month columns that contain integer values
            df["Day"] = df.Date_of_Journey.dt.day
            df["Month"] = df.Date_of_Journey.dt.month

            # Dropping "Date_of_Journey" column as it is not required anymore
            df.drop(columns=["Date_of_Journey"], inplace=True)

            # Change data type of Dep_Time from object to datetime
            df.Dep_Time = pd.to_datetime(df.Dep_Time)   

            # Creating two seperate columns named "Dep_hr" and "Dep_min" that contain integer values
            df["Dep_hr"] = df.Dep_Time.dt.hour
            df["Dep_min"] = df.Dep_Time.dt.minute 

            # Dropping "Dep_Time" column as it is not required anymore
            df.drop(columns=["Dep_Time"], inplace=True)

            # Change data type of Arrival_Time from object to datetime
            df.Arrival_Time = pd.to_datetime(df.Arrival_Time)

            # create two seperate columnsnamed "Arrival_hr" and "Arrival_min" that contain integer values
            df["Arrival_hr"] = df.Arrival_Time.dt.hour
            df["Arrival_min"] = df.Arrival_Time.dt.minute

            # Dropping "Arrival_Time" column as it is not required anymore
            df.drop(columns=["Arrival_Time"], inplace=True) 

            # we can convert all the values in Duration column into equivallent value in min
            def duration_in_min(dur):
                tt = 0
                for i in dur.split():
                    if 'h' in i:
                        tt = tt + int(i[:-1]) * 60
                    else:
                        tt = tt + int(i[:-1])
                return tt

            df.Duration = df.Duration.apply(duration_in_min)

            return df

        except Exception as e:
            logging.debug(str(e))
            raise FlightException(error_message=e, error_detail=sys)


    # Function to encode categorical features
    def encode_categorical_features(self, df:pd.DataFrame)->pd.DataFrame:
        """
        This function will encode categorical features

        df: train.csv/test.csv
        ================================================
        returns Pandas Dataframe
        """
        try:
            # Label Encoding
            label_encoder = LabelEncoder()
            # Encoding "Total_Stops" using LabelEncoder
            df.Total_Stops = label_encoder.fit_transform(df["Total_Stops"])

            # Encoding "Source", "Destination", "Airline" using OneHotEncoder
            df = pd.get_dummies(df, columns=["Source","Destination","Airline"])

            return df

        except Exception as e:
            logging.debug(str(e))
            raise FlightException(error_message=e, error_detail=sys)

    
    # Function to encde categorical features of each Airline DataFrame
    def encode_airline_df(self, df:pd.DataFrame)->pd.DataFrame:
        """
        This function will encode categorical features of each airline dataframe

        df: pd.DataFrame
        =======================================================================
        returns Pandas Dataframe
        """
        try:
            # drop unnecessary features
            df.drop(columns=["Airline"], inplace=True)

            # Encoding "Source", "Destination" using OneHotEncoder
            df = pd.get_dummies(df, columns=["Source","Destination"])

            # Label Encoding
            label_encoder = LabelEncoder()
            # Encoding "Total_Stops" using LabelEncoder
            df.Total_Stops = label_encoder.fit_transform(df["Total_Stops"])

            return df

        except Exception as e:
            logging.debug(str(e))
            raise FlightException(error_message=e, error_detail=sys)


    def initiate_data_transformation(self)->artifact_entity.DataTransformationArtifact:

        try:
            # Reading train_df and test_df file
            logging.info(f"Reading train dataframe")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info(f"Reading test dataframe")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info(f"Columns Before Transformation in Train Data: {train_df.columns}")
            logging.info(f"Columns Before Transformation in Test Data: {test_df.columns}")

            useless_features = self.data_transformation_config.useless_features
            logging.info(f"Dropping these {useless_features} features")

            for feature in useless_features:
                if feature in train_df.columns:
                    logging.info(f"{feature} removed from train_df")
                    train_df.drop(columns=[feature],inplace=True)

                if feature in test_df.columns:
                    logging.info(f"{feature} removed from test_df")
                    test_df.drop(columns=[feature],inplace=True)

            # Dropping NaN values
            logging.info("Dropping NaN values")
            train_df.dropna(inplace=True)
            test_df.dropna(inplace=True)

            # Transforming Date, Time and Duration feature in train_df and test_df
            logging.info("Transforming Date, Time and Duration feature in train_df and test_df")
            train_df = self.transform_datetime_features(df=train_df)
            test_df = self.transform_datetime_features(df=test_df)

            # Insert row if any category is missing in Categorical Features
            for col in ["Source", "Destination", "Airline"]:

                if len(train_df[col].unique()) > len(test_df[col].unique()):

                    for i in set(train_df[col].unique()).difference(list(test_df[col].unique())):

                        test_df.loc[len(test_df.index)] = [i,'Kolkata', 'Banglore', 120, '0', 12, 8, 8, 30, 10, 30, 6000]
                        logging.info("1 row Inserted")

            logging.info(f"new inserted rows: \n{test_df.iloc[-3:]}")

            airline_train_df = train_df.copy()
            airline_test_df = test_df.copy()

            # making Data for individual model
            logging.info("Making Dataset for each Airline")

            train_df_JetAirways = airline_train_df[airline_train_df.Airline == 'Jet Airways']
            train_df_JetAirways = self.encode_airline_df(df=train_df_JetAirways)
            test_df_JetAirways = airline_test_df[airline_test_df.Airline == 'Jet Airways']
            test_df_JetAirways = self.encode_airline_df(df=test_df_JetAirways)

            train_df_Indigo = airline_train_df[airline_train_df.Airline == 'IndiGo']
            train_df_Indigo = self.encode_airline_df(df=train_df_Indigo)
            test_df_Indigo = airline_test_df[airline_test_df.Airline == 'IndiGo']
            test_df_Indigo = self.encode_airline_df(df=test_df_Indigo)

            train_df_AirIndia = airline_train_df[airline_train_df.Airline == 'Air India']
            train_df_AirIndia = self.encode_airline_df(df=train_df_AirIndia)
            test_df_AirIndia = airline_test_df[airline_test_df.Airline == 'Air India']
            test_df_AirIndia = self.encode_airline_df(df=test_df_AirIndia)

            train_df_MultipleCarriers = airline_train_df[airline_train_df.Airline == 'Multiple carriers']
            train_df_MultipleCarriers = self.encode_airline_df(df=train_df_MultipleCarriers)
            test_df_MultipleCarriers = airline_test_df[airline_test_df.Airline == 'Multiple carriers']
            test_df_MultipleCarriers = self.encode_airline_df(df=test_df_MultipleCarriers)

            train_df_SpiceJet = airline_train_df[airline_train_df.Airline == 'SpiceJet']
            train_df_SpiceJet = self.encode_airline_df(df=train_df_SpiceJet)
            test_df_SpiceJet = airline_test_df[airline_test_df.Airline == 'SpiceJet']
            test_df_SpiceJet = self.encode_airline_df(df=test_df_SpiceJet)

            train_df_Vistara = airline_train_df[airline_train_df.Airline == 'Vistara']
            train_df_Vistara = self.encode_airline_df(df=train_df_Vistara)
            test_df_Vistara = airline_test_df[airline_test_df.Airline == 'Vistara']
            test_df_Vistara = self.encode_airline_df(df=test_df_Vistara)

            train_df_AirAsia = airline_train_df[airline_train_df.Airline == 'Air Asia']
            train_df_AirAsia = self.encode_airline_df(df=train_df_AirAsia)
            test_df_AirAsia = airline_test_df[airline_test_df.Airline == 'Air Asia']
            test_df_AirAsia = self.encode_airline_df(df=test_df_AirAsia)

            train_df_GoAir = airline_train_df[airline_train_df.Airline == 'GoAir']
            train_df_GoAir = self.encode_airline_df(df=train_df_GoAir)
            test_df_GoAir = airline_test_df[airline_test_df.Airline == 'GoAir']
            test_df_GoAir = self.encode_airline_df(df=test_df_GoAir)

            logging.info("Encoding All categorical features in train_df and test_df")
            train_df = self.encode_categorical_features(df=train_df)
            test_df = self.encode_categorical_features(df=test_df)

            logging.info("Create general_model folder in data_transformation folder if not available")
            general_model_dir = os.path.dirname(self.data_transformation_config.transform_train_path)
            os.makedirs(general_model_dir, exist_ok=True)

            logging.info("Save train_df and test_df to general_model folder")
            train_df.to_csv(path_or_buf=self.data_transformation_config.transform_train_path, index=False, header=True)
            test_df.to_csv(path_or_buf=self.data_transformation_config.transform_test_path, index=False, header=True)

            logging.info("Create JetAirways_model folder in data_transformation folder if not available")
            jetairways_model_dir = os.path.dirname(self.data_transformation_config.JetAirways_train_path)
            os.makedirs(jetairways_model_dir, exist_ok=True)

            logging.info("Save train_df_JetAirways and test_df_JetAirways to JetAirways_model folder")
            train_df_JetAirways.to_csv(path_or_buf=self.data_transformation_config.JetAirways_train_path, index=False, header=True)
            test_df_JetAirways.to_csv(path_or_buf=self.data_transformation_config.JetAirways_test_path, index=False, header=True)

            logging.info("Create Indigo_model folder in data_transformation folder if not available")
            indigo_model_dir = os.path.dirname(self.data_transformation_config.Indigo_train_path)
            os.makedirs(indigo_model_dir, exist_ok=True)

            logging.info("Save train_df_Indigo and test_df_Indigo to Indigo_model folder")
            train_df_Indigo.to_csv(path_or_buf=self.data_transformation_config.Indigo_train_path, index=False, header=True)
            test_df_Indigo.to_csv(path_or_buf=self.data_transformation_config.Indigo_test_path, index=False, header=True)

            logging.info("Create AirIndia_model folder in data_transformation folder if not available")
            airindia_model_dir = os.path.dirname(self.data_transformation_config.AirIndia_train_path)
            os.makedirs(airindia_model_dir, exist_ok=True)

            logging.info("Save train_df_AirIndia and test_df_AirIndia to AirIndia_model folder")
            train_df_AirIndia.to_csv(path_or_buf=self.data_transformation_config.AirIndia_train_path, index=False, header=True)
            test_df_AirIndia.to_csv(path_or_buf=self.data_transformation_config.AirIndia_test_path, index=False, header=True)

            logging.info("Create MultipleCarriers_model folder in data_transformation folder if not available")
            multiplecarries_model_dir = os.path.dirname(self.data_transformation_config.MultipleCarriers_train_path)
            os.makedirs(multiplecarries_model_dir, exist_ok=True)

            logging.info("Save train_df_MultipleCarriers and test_df_MultipleCarriers to MultipleCarriers_model folder")
            train_df_MultipleCarriers.to_csv(path_or_buf=self.data_transformation_config.MultipleCarriers_train_path, index=False, header=True)
            test_df_MultipleCarriers.to_csv(path_or_buf=self.data_transformation_config.MultipleCarriers_test_path, index=False, header=True)

            logging.info("Create SpiceJet_model folder in data_transformation folder if not available")
            spicejet_model_dir = os.path.dirname(self.data_transformation_config.SpiceJet_train_path)
            os.makedirs(spicejet_model_dir, exist_ok=True)

            logging.info("Save train_df_SpiceJet and test_df_SpiceJet to SpiceJet_model folder")
            train_df_SpiceJet.to_csv(path_or_buf=self.data_transformation_config.SpiceJet_train_path, index=False, header=True)
            test_df_SpiceJet.to_csv(path_or_buf=self.data_transformation_config.SpiceJet_test_path, index=False, header=True)

            logging.info("Create Vistara_model folder in data_transformation folder if not available")
            vistara_model_dir = os.path.dirname(self.data_transformation_config.Vistara_train_path)
            os.makedirs(vistara_model_dir, exist_ok=True)

            logging.info("Save train_df_Vistara and test_df_Vistara to Vistara_model folder")
            train_df_Vistara.to_csv(path_or_buf=self.data_transformation_config.Vistara_train_path, index=False, header=True)
            test_df_Vistara.to_csv(path_or_buf=self.data_transformation_config.Vistara_test_path, index=False, header=True)

            logging.info("Create AirAsia_model folder in data_transformation folder if not available")
            airasia_model_dir = os.path.dirname(self.data_transformation_config.AirAsia_train_path)
            os.makedirs(airasia_model_dir, exist_ok=True)

            logging.info("Save train_df_AirAsia and test_df_AirAsia to AirAsia_model folder")
            train_df_AirAsia.to_csv(path_or_buf=self.data_transformation_config.AirAsia_train_path, index=False, header=True)
            test_df_AirAsia.to_csv(path_or_buf=self.data_transformation_config.AirAsia_test_path, index=False, header=True)

            logging.info("Create GoAir_model folder in data_transformation folder if not available")
            goair_model_dir = os.path.dirname(self.data_transformation_config.GoAir_train_path)
            os.makedirs(goair_model_dir, exist_ok=True)

            logging.info("Save train_df_GoAir and test_df_GoAir to GoAir_model folder")
            train_df_GoAir.to_csv(path_or_buf=self.data_transformation_config.GoAir_train_path, index=False, header=True)
            test_df_GoAir.to_csv(path_or_buf=self.data_transformation_config.GoAir_test_path, index=False, header=True)

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_train_path=self.data_transformation_config.transform_train_path, 
                transform_test_path=self.data_transformation_config.transform_test_path, 
                JetAirways_train_path=self.data_transformation_config.JetAirways_train_path, 
                JetAirways_test_path=self.data_transformation_config.JetAirways_test_path, 
                Indigo_train_path=self.data_transformation_config.Indigo_train_path, 
                Indigo_test_path=self.data_transformation_config.Indigo_test_path, 
                AirIndia_train_path=self.data_transformation_config.AirIndia_train_path, 
                AirIndia_test_path=self.data_transformation_config.AirIndia_test_path, 
                MultipleCarriers_train_path=self.data_transformation_config.MultipleCarriers_train_path, 
                MultipleCarriers_test_path=self.data_transformation_config.MultipleCarriers_test_path, 
                SpiceJet_train_path=self.data_transformation_config.SpiceJet_train_path, 
                SpiceJet_test_path=self.data_transformation_config.SpiceJet_test_path, 
                Vistara_train_path=self.data_transformation_config.Vistara_train_path, 
                Vistara_test_path=self.data_transformation_config.Vistara_test_path,
                AirAsia_train_path=self.data_transformation_config.AirAsia_train_path, 
                AirAsia_test_path=self.data_transformation_config.AirAsia_test_path, 
                GoAir_train_path=self.data_transformation_config.GoAir_train_path, 
                GoAir_test_path=self.data_transformation_config.GoAir_test_path
            )

            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            logging.debug(str(e))
            raise FlightException(error_message=e, error_detail=sys)

