from flight import utils
from flight.entity import config_entity
from flight.entity import artifact_entity
from flight.exception import FlightException
from flight.logger import logging
import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataIngestion:
    
    def __init__(self,data_ingestion_config:config_entity.DataIngestionConfig):

        try:
            logging.info(f"{'>>'*20} Data Ingestion {'<<'*20}")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            logging.debug(str(e))
            raise FlightException(error_message=e, error_detail=sys)


    def initiate_data_ingestion(self)->artifact_entity.DataIngestionArtifact:

        try:
            logging.info(f"Exporting Collection Data as Pandas Dataframe")
            # Exporting Collection Data as Pandas Dataframe
            df:pd.DataFrame = utils.get_collection_as_dataframe(
                database_name = self.data_ingestion_config.database_name,
                collection_name = self.data_ingestion_config.collection_name
            )

            logging.info("Save Data into data_ingestion directory")

            # Drop NA values
            logging.info("Dropping rows from dataframe when NaN value present")
            df.dropna(inplace=True)

            # Save Data in feature_store folder
            logging.info("Create feature_store folder if not available")
            # Create feature_store folder if not available
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir,exist_ok=True)

            logging.info("Save DF to feature_store folder")
            # Save DF to feature_store folder
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path,index=False,header=True)

            logging.info("Split dataset into train and test set")
            # Split dataset into train and test set
            train_df, test_df = train_test_split(df,test_size=self.data_ingestion_config.test_size,random_state=42)

            # Save Data in dataset folder
            logging.info("Create dataset folder if not available")
            # Create dataset folder if not available
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir,exist_ok=True)

            logging.info("Save train.csv and test.csv to dataset folder")
            # Save train.csv and test.csv to dataset folder
            df.to_csv(path_or_buf = self.data_ingestion_config.train_file_path, index=False, header=True)
            df.to_csv(path_or_buf = self.data_ingestion_config.test_file_path, index=False, header=True)

            logging.info("Preparing flight.csv, train.csv, test.csv")
            # Preparing Artifact
            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path = self.data_ingestion_config.feature_store_file_path,
                train_file_path = self.data_ingestion_config.train_file_path,
                test_file_path = self.data_ingestion_config.test_file_path
            )

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")

            return data_ingestion_artifact

        except Exception as e:
            logging.debug(str(e))
            raise FlightException(error_message=e, error_detail=sys)

