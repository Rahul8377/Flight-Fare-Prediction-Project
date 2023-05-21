from flight.logger import logging
from flight.exception import FlightException
from flight.entity import config_entity
from flight.components.data_ingestion import DataIngestion
from flight.components.data_validation import DataValidation
from flight.components.data_transformation import DataTransformation
from flight.components.model_training import ModelTrainer
from flight.components.model_pusher import ModelPusher

import sys,os

if __name__ == "__main__":
    try:
        # Training Pipeline
        training_pipeline_config = config_entity.TrainingPipelineConfig()

        # Data Ingestion
        data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config= data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        # Data Validation
        data_validation_config = config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
        data_validation_artifact = data_validation.initiate_data_validation()
          
        # Data Transformation
        data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation = DataTransformation(data_transformation_config=data_transformation_config, data_ingestion_artifact=data_ingestion_artifact)
        data_transformation_artifact = data_transformation.initiate_data_transformation()         

        # Model Training
        model_training_config = config_entity.ModelTrainingConfig(training_pipeline_config=training_pipeline_config)
        model_training = ModelTrainer(model_training_config=model_training_config, data_transformation_artifact=data_transformation_artifact)
        model_training_artifact = model_training.initiate_model_trainer()
          
        # Model Pusher
        model_pusher_config = config_entity.ModelPusherConfig()
        model_pusher = ModelPusher(model_pusher_config=model_pusher_config, model_training_artifact=model_training_artifact)
        model_pusher_artifact = model_pusher.initiate_model_pusher()

    except Exception as e:
          logging.debug(str(e))
          raise FlightException(e, sys)