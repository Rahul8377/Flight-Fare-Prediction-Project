from flight.entity import config_entity
from flight.entity import artifact_entity
from flight.exception import FlightException
from flight.logger import logging
from typing import Optional
from scipy.stats import ks_2samp
from flight import utils
import os, sys
import pandas as pd 
import numpy as np


class DataValidation:

    def __init__(self,
                    data_validation_config:config_entity.DataValidationConfig,
                    data_ingestion_artifact:artifact_entity.DataIngestionArtifact):

        try:
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = dict()

        except Exception as e:
            logging.debug(str(e))
            raise FlightException(error_message=e, error_detail=sys)


    # Function to drop column which contains number of null value greater then threshold limit
    def drop_missing_values_columns(self, df:pd.DataFrame, report_key_name:str)->Optional[pd.DataFrame]:
        """
        This function will drop column which contains missing value more than specified threshold

        df: Accepts a pandas dataframe
        ====================================================================================================
        returns Pandas Dataframe if atleast a single column is available after missing column drop else None
        """
        try:
            threshold = self.data_validation_config.missing_threshold
            null_report = df.isnull().sum() / df.shape[0]

            # Selecting Column name which contains Null Value
            logging.info(f"Selecting Column name which conains Null value above thresold value {threshold}")
            drop_column_names = null_report[null_report > threshold].index

            logging.info(f"Columns to drop: {list(drop_column_names)}")
            self.validation_error[report_key_name] = list(drop_column_names)
            df.drop(list(drop_column_names), axis=1, inplace=True)

            # return None if no columns left
            if len(df.columns) == 0:
                return None

            return df

        except Exception as e:
            logging.debug(str(e))
            raise FlightException(error_message=e, error_detail=sys)


    # Function to validate count the number of columns
    def is_required_column_exist(self, base_df:pd.DataFrame, current_df:pd.DataFrame, report_key_name:str)->bool:
        """
        This function will check if the number of columns in base_df and current_df is same or not

        base_df: Base DataFrame(Data_Train.xlsx)
        current_df: Current DataFrame(train.csv/test.csv)
        ============================================================================================
        returns True if number of columns in base_df and current_df are same otherwise returns False
        """
        try:
            base_cloumns = base_df.columns
            current_columns = current_df.columns

            missing_columns = []
            for column in base_cloumns:
                if column not in current_columns:
                    logging.info(f"Column: [{column}] is not available")
                    missing_columns.append(column)

            if len(missing_columns) > 0:
                self.validation_error[report_key_name] = missing_columns
                return False

            return True

        except Exception as e:
            logging.debug(str(e))
            raise FlightException(error_message=e, error_detail=sys)



    def data_drift(self, base_df:pd.DataFrame, current_df:pd.DataFrame, report_key_name:str):
        """
        This function will make data drift report(dictionary format) based on p-value

        base_df: Base DataFrame(Data_Train.xlsx)
        current_df: Current DataFrame(train.csv/test.csv)
        ====================================================================================
        This function does not return anything
        """
        try:
            drift_report = dict()

            price_cloumn = "Price"
            
            base_data, current_data = base_df[price_cloumn], current_df[price_cloumn]

            # Null Hypothesis is that both column data drawn from same distribution
            logging.info(f"Hypothesis {price_cloumn} : {base_data.dtype}, {current_data.dtype}")
            same_distribution = ks_2samp(base_data,current_data)

            if same_distribution.pvalue > 0.5:

                # We are accepting Null Hypothesis
                drift_report[price_cloumn] = {
                    "p-value": float(same_distribution.pvalue),
                    "same_distribution": True
                }

            else:

                # We are rejecting Null Hypothesis
                drift_report[price_cloumn] = {
                    "p-value": float(same_distribution.pvalue),
                    "same_distribution": False
                }

            self.validation_error[report_key_name] = drift_report
        
        except Exception as e:
            logging.debug(str(e))
            raise FlightException(error_message=e, error_detail=sys)


    def is_category_value_exist(self, base_df:pd.DataFrame, current_df:pd.DataFrame, report_key_name:str)->bool:
        """
        This function will check if category value is exist in both base_df and current_df

        base_df: Base DataFrame(Data_Train.xlsx)
        current_df: Current DataFrame(train.csv/test.csv)
        ====================================================================================
        returns True if category in base_df and current_df are same otherwise returns false
        """
        try:
            missing_cat = dict()
            cat_nom_columns = ["Airline", "Source", "Destination"]

            for column in cat_nom_columns:
                base_column_unique = base_df[column].unique()
                current_column_unique = current_df[column].unique()

                if len(base_column_unique) >= len(current_column_unique):

                    missing_values = []

                    for val in base_column_unique:
                        if val not in current_column_unique:
                            missing_values.append(val)

                    missing_cat[column] = missing_values
                    logging.info(f"In {column} column missing categories are {missing_values}")

                elif len(base_column_unique) < len(current_column_unique):

                    missing_values = []

                    for val in base_column_unique:
                        if val not in current_column_unique:
                            missing_values.append(val)

                    missing_cat[column] = missing_values
                    logging.info(f"In {column} column excess categories are {missing_values}")

            self.validation_error[report_key_name] = missing_cat

        except Exception as e:
            logging.debug(str(e))
            raise FlightException(error_message=e, error_detail=sys)            

        
    def initiate_data_validation(self)->artifact_entity.DataValidationArtifact:

        try:

            # Reading Data_Train.xlsx file
            logging.info("Reading Base DataFrame")
            base_df = pd.read_excel(self.data_validation_config.base_file_path)

            # Dropping column from base_df which contains Null value more than threshold value 
            logging.info("Dropping column from base_df which contains Null value more than threshold value")
            base_df = self.drop_missing_values_columns(df=base_df, report_key_name="missing_values_column_within_base_dataset")

            # Reading train_df and test_df file
            logging.info(f"Reading train dataframe")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info(f"Reading test dataframe")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # dropping column from train_df and test_df which contains Null value more than threshold value 
            logging.info("Dropping column from train_df")
            train_df = self.drop_missing_values_columns(df=train_df, report_key_name="missing_values_column_within_train_dataset")
            logging.info("Dropping column from test_df")
            test_df = self.drop_missing_values_columns(df=test_df, report_key_name="missing_values_column_within_test_dataset")

            # check no of column in base_df and train_df
            logging.info(f"Is all required columns are present in train_df")
            train_df_column_status = self.is_required_column_exist(base_df=base_df, current_df=train_df, report_key_name="missing_column_within_train_dataset")

            # check no of column in base_df and test_df
            logging.info(f"Is all required columns are present in test_df")
            test_df_column_status = self.is_required_column_exist(base_df=base_df, current_df=test_df, report_key_name="missing_column_within_test_dataset")

            if train_df_column_status:
                logging.info(f"As all column are available in train df hence detecting data drift")
                self.data_drift(base_df=base_df, current_df=train_df, report_key_name="data_drift_within_train_dataset")
                logging.info("Now checking for value exist in categorical feature of train_df and base_df")
                self.is_category_value_exist(base_df=base_df, current_df=train_df, report_key_name="categorical_value_exist_in_traindf")

            if test_df_column_status:
                logging.info(f"As all column are available in test df hence detecting data drift")
                self.data_drift(base_df=base_df, current_df=test_df, report_key_name="data_drift_within_test_dataset")
                logging.info("Now checking for value exist in categorical feature of test_df and base_df")
                self.is_category_value_exist(base_df=base_df, current_df=test_df, report_key_name="categorical_value_exist_in_testdf")

            # write the Report
            logging.info("write report in yaml file")
            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path, data=self.validation_error)

            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path)
            logging.info(f"Data Validation Artifact : {data_validation_artifact}")
            return data_validation_artifact
            
        
        except Exception as e:
            logging.debug(str(e))
            raise FlightException(error_message=e, error_detail=sys)