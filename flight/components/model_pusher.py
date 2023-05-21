from flight.entity import config_entity
from flight.entity import artifact_entity
from flight.exception import FlightException
from flight.logger import logging
from flight.utils import load_object, save_object
import os,sys

class ModelPusher:

    def __init__(self,
                    model_pusher_config:config_entity.ModelPusherConfig,
                    model_training_artifact:artifact_entity.ModelTrainingArtifact):

        try:
            logging.info(f"{'>>'*20} Model Pusher {'<<'*20}")
            self.model_pusher_config = model_pusher_config
            self.model_training_artifact = model_training_artifact

        except Exception as e:
            logging.debug(str(e))
            raise FlightException(error_message=e, error_detail=sys)


    def initiate_model_pusher(self)->artifact_entity.ModelPusherArtifact:

        try:
            # load Models
            logging.info("Loading Models from Model Training Artifact")
            general_model = load_object(file_path=self.model_training_artifact.gen_model_path)
            jetairways_model = load_object(file_path=self.model_training_artifact.JetAirways_model_path)
            indigo_model = load_object(file_path=self.model_training_artifact.Indigo_model_path)
            airindia_model = load_object(file_path=self.model_training_artifact.AirIndia_model_path)
            multiplecarriers_model = load_object(file_path=self.model_training_artifact.MultipleCarriers_model_path)
            spicejet_model = load_object(file_path=self.model_training_artifact.SpiceJet_model_path)
            vistara_model = load_object(file_path=self.model_training_artifact.Vistara_model_path)
            airasia_model = load_object(file_path=self.model_training_artifact.AirAsia_model_path)
            goair_model = load_object(file_path=self.model_training_artifact.GoAir_model_path)

            
            # Check if any directory is alreay exist or not in saved_model directory
            logging.info("Checking if any directory is alreay exist or not in saved_model directory")
            list_dir = os.listdir(self.model_pusher_config.saved_model_dir)
            if len(list_dir) == 0:
                saved_model_path = os.path.join(self.model_pusher_config.saved_model_dir,f"{0}")
            else:
                latest_dir_num = int(max(list_dir)) + 1
                saved_model_path = os.path.join(self.model_pusher_config.saved_model_dir,f"{latest_dir_num}")

            # Save Models
            logging.info("Saving Models in saved_model directory")
            save_object(file_path=os.path.join(saved_model_path,"gen_model.pkl"), obj=general_model)
            save_object(file_path=os.path.join(saved_model_path,"JetAirways_model.pkl"), obj=jetairways_model)
            save_object(file_path=os.path.join(saved_model_path,"Indigo_model.pkl"), obj=indigo_model)
            save_object(file_path=os.path.join(saved_model_path,"AirIndia_model.pkl"), obj=airindia_model)
            save_object(file_path=os.path.join(saved_model_path,"MultipleCarriers_model.pkl"), obj=multiplecarriers_model)
            save_object(file_path=os.path.join(saved_model_path,"SpiceJet_model.pkl"), obj=spicejet_model)
            save_object(file_path=os.path.join(saved_model_path,"Vistara_model.pkl"), obj=vistara_model)
            save_object(file_path=os.path.join(saved_model_path,"AirAsia_model.pkl"), obj=airasia_model)
            save_object(file_path=os.path.join(saved_model_path,"GoAir_model.pkl"), obj=goair_model)

            # Prepare Artifact
            model_pusher_artifact = artifact_entity.ModelPusherArtifact(saved_model_dir=saved_model_path)
            logging.info(f"Model Pusher Artifact: {model_pusher_artifact}")

            return model_pusher_artifact


        except Exception as e:
            logging.debug(str(e))
            raise FlightException(error_message=e, error_detail=sys)