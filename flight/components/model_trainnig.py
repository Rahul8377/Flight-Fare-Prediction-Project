from flight.entity import config_entity
from flight.entity import artifact_entity
from flight.exception import FlightException
from flight.logger import logging
from flight import utils
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from flight import utils
import os,sys
import pandas as pd 
import numpy as np

class ModelTrainer:

    def __init__(self, 
                    model_training_config:config_entity.ModelTrainingConfig,
                    data_transformation_artifact:artifact_entity.DataTransformationArtifact):

        try:
            logging.info(f"{'>>'*20} Model Training {'<<'*20}")
            self.model_training_config = model_training_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            logging.debug(str(e))
            raise FlightException(error_message=e, error_detail=sys)


    def fine_tune(self, model, X, y)->dict:
        """
        This function is used to choose best parameter

        model: the model on which best parameter is applicable
        X: Independent Feature
        y: Dependent Feature
        ========================================================
        return best parameters in dictionary format
        """
        try:
            grid_search = GridSearchCV(estimator=model, param_grid=self.model_training_config.grid_param)
            grid_search.fit(X,y)
            best_param = grid_search.best_params_
            logging.info(f"Best Parameter = {best_param}")
            return best_param

        except Exception as e:
            logging.debug(str(e))
            raise FlightException(error_message=e, error_detail=sys)


    def train_model(self, X, y):
        """
        This function is used to train model

        X: Independent feature
        y: dependent feature
        ============================================
        return: random_forest_regressor
        """
        try:
            random_forest_regressor = RandomForestRegressor()
            # random_forest_regressor = RandomForestRegressor()
            random_forest_regressor.fit(X, y)
            return random_forest_regressor
        
        except Exception as e:
            logging.debug(str(e))
            raise FlightException(error_message=e, error_detail=sys)


    def initiate_model_trainer(self)->artifact_entity.ModelTrainingArtifact:
        try:
            TARGET_COLUMN = self.model_training_config.TARGET_COLUMN

            logging.info(f"Loading General_Model train and test data set")
            gen_train_data = pd.read_csv(self.data_transformation_artifact.transform_train_path)
            gen_test_data = pd.read_csv(self.data_transformation_artifact.transform_test_path)

            logging.info(f"Splitting input and target feature from both train and test data")
            X_train, y_train = gen_train_data.drop(TARGET_COLUMN, axis=1), gen_train_data[TARGET_COLUMN]
            X_test, y_test = gen_test_data.drop(TARGET_COLUMN, axis=1), gen_test_data[TARGET_COLUMN]

            # logging.info(f"Fine Tune The Model")
            # random_forest_regressor = RandomForestRegressor()
            # best_param = self.fine_tune(random_forest_regressor, X=X_train, y=y_train)

            logging.info(f"Train The General Model")
            model = self.train_model(X=X_train,y=y_train)

            logging.info("Predicting General Model")
            y_pred = model.predict(X_test)

            # General Model Performance
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            logging.info(f"General Model: Train Score={train_score} and Test Score={test_score}")
            R2_score = metrics.r2_score(y_test, y_pred)
            logging.info(f"R2_Score of General Model: {R2_score}")

            # Check for Model is Good or Bad and Overfitting Condition
            logging.info("Checkinng General Model is Good or Bad")
            if R2_score < self.model_training_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}: model actual score is {R2_score}")

            logging.info(f"Checking General Model is Overfitting or not")
            diff = abs(train_score - test_score)
            if diff > self.model_training_config.overfitting_threshold:
                raise Exception(f"Train and test Score difference: {diff} is more than Overfitting threshold {self.model_training_config.overfitting_threshold}")
            
            # Save the trained model
            logging.info(f"Saving General Model Object")
            utils.save_object(file_path=self.model_training_config.gen_model_path,obj=model)
            logging.info(f"{'-'*30}")

            # Jetairways
            logging.info(f"Loading Train and Test data of JetAirways Model")
            jetairways_train_data = pd.read_csv(self.data_transformation_artifact.JetAirways_train_path)
            jetairways_test_data = pd.read_csv(self.data_transformation_artifact.JetAirways_test_path)

            X_train_jetairways, y_train_jetairways = jetairways_train_data.drop(TARGET_COLUMN, axis=1), jetairways_train_data[TARGET_COLUMN]
            X_test_jetairways, y_test_jetairways = jetairways_test_data.drop(TARGET_COLUMN, axis=1), jetairways_test_data[TARGET_COLUMN]

            logging.info("Training the JetAirways Model")
            jetairways_model = self.train_model(X=X_train_jetairways, y=y_train_jetairways)
            
            logging.info("Predicting JetAirways Model")
            y_pred_jetairways = jetairways_model.predict(X_test_jetairways)

            train_score_jetairways = jetairways_model.score(X_train_jetairways, y_train_jetairways)
            test_score_jetairways = jetairways_model.score(X_test_jetairways, y_test_jetairways)
            logging.info(f"JetAirways Model: Train Score={train_score_jetairways} and Test Score={test_score_jetairways}")
            R2_score_jetairways = metrics.r2_score(y_test_jetairways, y_pred_jetairways)
            logging.info(f"R2_Score of JetAirways Model: {R2_score_jetairways}")

            logging.info("Checking JetAirways Model is Good or Bad")
            if R2_score_jetairways < self.model_training_config.expected_score:
                raise Exception(f"JetAirways Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}:JetAirways model actual score is {R2_score}")

            logging.info(f"Checking JetAirways Model is Overfitting or not")
            diff = abs(train_score_jetairways - test_score_jetairways)
            if diff > self.model_training_config.overfitting_threshold:
                raise Exception(f"JetAirways Train and JetAirways Test Score difference: {diff} is more than Overfitting threshold {self.model_training_config.overfitting_threshold}")

            logging.info(f"Saving JetAirways Model Object")
            utils.save_object(file_path=self.model_training_config.JetAirways_model_path,obj=jetairways_model)
            logging.info(f"{'-'*30}")

        
            # Indigo
            logging.info(f"Loading Train and Test data of Indigo Model")
            indigo_train_data = pd.read_csv(self.data_transformation_artifact.Indigo_train_path)
            indigo_test_data = pd.read_csv(self.data_transformation_artifact.Indigo_test_path)

            X_train_indigo, y_train_indigo = indigo_train_data.drop(TARGET_COLUMN, axis=1), indigo_train_data[TARGET_COLUMN]
            X_test_indigo, y_test_indigo = indigo_test_data.drop(TARGET_COLUMN, axis=1), indigo_test_data[TARGET_COLUMN]

            logging.info("Training the indigo Model")
            indigo_model = self.train_model(X=X_train_indigo, y=y_train_indigo)
            
            logging.info("Predicting indigo Model")
            y_pred_indigo = indigo_model.predict(X_test_indigo)

            train_score_indigo = indigo_model.score(X_train_indigo, y_train_indigo)
            test_score_indigo = indigo_model.score(X_test_indigo, y_test_indigo)
            logging.info(f"indigo Model: Train Score={train_score_indigo} and Test Score={test_score_indigo}")
            R2_score_indigo = metrics.r2_score(y_test_indigo, y_pred_indigo)
            logging.info(f"R2_Score of indigo Model: {R2_score_indigo}")

            logging.info("Checking indigo Model is Good or Bad")
            if R2_score_indigo < self.model_training_config.expected_score:
                raise Exception(f"indigo Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}:indigo model actual score is {R2_score}")

            logging.info(f"Checking indigo Model is Overfitting or not")
            diff = abs(train_score_indigo - test_score_indigo)
            if diff > self.model_training_config.overfitting_threshold:
                raise Exception(f"indigo Train and indigo Test Score difference: {diff} is more than Overfitting threshold {self.model_training_config.overfitting_threshold}")

            logging.info(f"Saving Indigo Model Object")
            utils.save_object(file_path=self.model_training_config.Indigo_model_path,obj=indigo_model)
            logging.info(f"{'-'*30}")

            
            # AirIndia
            logging.info(f"Loading Train and Test data of AirIndia Model")
            airindia_train_data = pd.read_csv(self.data_transformation_artifact.AirIndia_train_path)
            airindia_test_data = pd.read_csv(self.data_transformation_artifact.AirIndia_test_path)

            X_train_airindia, y_train_airindia = airindia_train_data.drop(TARGET_COLUMN, axis=1), airindia_train_data[TARGET_COLUMN]
            X_test_airindia, y_test_airindia = airindia_test_data.drop(TARGET_COLUMN, axis=1), airindia_test_data[TARGET_COLUMN]

            logging.info("Training the airindia Model")
            airindia_model = self.train_model(X=X_train_airindia, y=y_train_airindia)
            
            logging.info("Predicting airindia Model")
            y_pred_airindia = airindia_model.predict(X_test_airindia)

            train_score_airindia = airindia_model.score(X_train_airindia, y_train_airindia)
            test_score_airindia = airindia_model.score(X_test_airindia, y_test_airindia)
            logging.info(f"airindia Model: Train Score={train_score_airindia} and Test Score={test_score_airindia}")
            R2_score_airindia = metrics.r2_score(y_test_airindia, y_pred_airindia)
            logging.info(f"R2_Score of airindia Model: {R2_score_airindia}")

            logging.info("Checking airindia Model is Good or Bad")
            if R2_score_airindia < self.model_training_config.expected_score:
                raise Exception(f"airindia Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}:airindia model actual score is {R2_score}")

            logging.info(f"Checking airindia Model is Overfitting or not")
            diff = abs(train_score_airindia - test_score_airindia)
            if diff > self.model_training_config.overfitting_threshold:
                raise Exception(f"airindia Train and airindia Test Score difference: {diff} is more than Overfitting threshold {self.model_training_config.overfitting_threshold}")

            logging.info(f"Saving AirIndia Model Object")
            utils.save_object(file_path=self.model_training_config.AirIndia_model_path,obj=airindia_model)
            logging.info(f"{'-'*30}")
            
            
            # MultipleCarriers
            logging.info(f"Loading Train and Test data of MultipleCarriers Model")
            multiplecarriers_train_data = pd.read_csv(self.data_transformation_artifact.MultipleCarriers_train_path)
            multiplecarriers_test_data = pd.read_csv(self.data_transformation_artifact.MultipleCarriers_test_path)

            X_train_multiplecarriers, y_train_multiplecarriers = multiplecarriers_train_data.drop(TARGET_COLUMN, axis=1), multiplecarriers_train_data[TARGET_COLUMN]
            X_test_multiplecarriers, y_test_multiplecarriers = multiplecarriers_test_data.drop(TARGET_COLUMN, axis=1), multiplecarriers_test_data[TARGET_COLUMN]

            logging.info("Training the multiplecarriers Model")
            multiplecarriers_model = self.train_model(X=X_train_multiplecarriers, y=y_train_multiplecarriers)
            
            logging.info("Predicting multiplecarriers Model")
            y_pred_multiplecarriers = multiplecarriers_model.predict(X_test_multiplecarriers)

            train_score_multiplecarriers = multiplecarriers_model.score(X_train_multiplecarriers, y_train_multiplecarriers)
            test_score_multiplecarriers = multiplecarriers_model.score(X_test_multiplecarriers, y_test_multiplecarriers)
            logging.info(f"multiplecarriers Model: Train Score={train_score_multiplecarriers} and Test Score={test_score_multiplecarriers}")
            R2_score_multiplecarriers = metrics.r2_score(y_test_multiplecarriers, y_pred_multiplecarriers)
            logging.info(f"R2_Score of multiplecarriers Model: {R2_score_multiplecarriers}")

            logging.info("Checking multiplecarriers Model is Good or Bad")
            if R2_score_multiplecarriers < self.model_training_config.expected_score:
                raise Exception(f"multiplecarriers Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}:multiplecarriers model actual score is {R2_score}")

            logging.info(f"Checking multiplecarriers Model is Overfitting or not")
            diff = abs(train_score_multiplecarriers - test_score_multiplecarriers)
            if diff > self.model_training_config.overfitting_threshold:
                raise Exception(f"multiplecarriers Train and multiplecarriers Test Score difference: {diff} is more than Overfitting threshold {self.model_training_config.overfitting_threshold}")
            
            logging.info(f"Saving Multiple_carries Model Object")
            utils.save_object(file_path=self.model_training_config.MultipleCarries_model_path,obj=multiplecarriers_model)
            logging.info(f"{'-'*30}")

            
            # SpiceJet
            logging.info(f"Loading Train and Test data of SpiceJet Model")
            spicejet_train_data = pd.read_csv(self.data_transformation_artifact.SpiceJet_train_path)
            spicejet_test_data = pd.read_csv(self.data_transformation_artifact.SpiceJet_test_path)

            X_train_spicejet, y_train_spicejet = spicejet_train_data.drop(TARGET_COLUMN, axis=1), spicejet_train_data[TARGET_COLUMN]
            X_test_spicejet, y_test_spicejet = spicejet_test_data.drop(TARGET_COLUMN, axis=1), spicejet_test_data[TARGET_COLUMN]

            logging.info("Training the spicejet Model")
            spicejet_model = self.train_model(X=X_train_spicejet, y=y_train_spicejet)
            
            logging.info("Predicting spicejet Model")
            y_pred_spicejet = spicejet_model.predict(X_test_spicejet)

            train_score_spicejet = spicejet_model.score(X_train_spicejet, y_train_spicejet)
            test_score_spicejet = spicejet_model.score(X_test_spicejet, y_test_spicejet)
            logging.info(f"spicejet Model: Train Score={train_score_spicejet} and Test Score={test_score_spicejet}")
            R2_score_spicejet = metrics.r2_score(y_test_spicejet, y_pred_spicejet)
            logging.info(f"R2_Score of spicejet Model: {R2_score_spicejet}")

            logging.info("Checking spicejet Model is Good or Bad")
            if R2_score_spicejet < self.model_training_config.expected_score:
                raise Exception(f"spicejet Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}:spicejet model actual score is {R2_score}")

            logging.info(f"Checking spicejet Model is Overfitting or not")
            diff = abs(train_score_spicejet - test_score_spicejet)
            if diff > self.model_training_config.overfitting_threshold:
                raise Exception(f"spicejet Train and spicejet Test Score difference: {diff} is more than Overfitting threshold {self.model_training_config.overfitting_threshold}")

            logging.info(f"Saving SpiceJet Model Object")
            utils.save_object(file_path=self.model_training_config.SpiceJet_model_path,obj=spicejet_model)
            logging.info(f"{'-'*30}")

            
            # Vistara
            logging.info(f"Loading Train and Test data of Vistara Model")
            vistara_train_data = pd.read_csv(self.data_transformation_artifact.Vistara_train_path)
            vistara_test_data = pd.read_csv(self.data_transformation_artifact.Vistara_test_path)

            X_train_vistara, y_train_vistara = vistara_train_data.drop(TARGET_COLUMN, axis=1), vistara_train_data[TARGET_COLUMN]
            X_test_vistara, y_test_vistara = vistara_test_data.drop(TARGET_COLUMN, axis=1), vistara_test_data[TARGET_COLUMN]

            logging.info("Training the vistara Model")
            vistara_model = self.train_model(X=X_train_vistara, y=y_train_vistara)
            
            logging.info("Predicting vistara Model")
            y_pred_vistara = vistara_model.predict(X_test_vistara)

            train_score_vistara = vistara_model.score(X_train_vistara, y_train_vistara)
            test_score_vistara = vistara_model.score(X_test_vistara, y_test_vistara)
            logging.info(f"vistara Model: Train Score={train_score_vistara} and Test Score={test_score_vistara}")
            R2_score_vistara = metrics.r2_score(y_test_vistara, y_pred_vistara)
            logging.info(f"R2_Score of vistara Model: {R2_score_vistara}")

            logging.info("Checking vistara Model is Good or Bad")
            if R2_score_vistara < self.model_training_config.expected_score:
                raise Exception(f"vistara Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}:vistara model actual score is {R2_score}")

            logging.info(f"Checking vistara Model is Overfitting or not")
            diff = abs(train_score_vistara - test_score_vistara)
            if diff > self.model_training_config.overfitting_threshold:
                raise Exception(f"vistara Train and vistara Test Score difference: {diff} is more than Overfitting threshold {self.model_training_config.overfitting_threshold}")

            logging.info(f"Saving Vistara Model Object")
            utils.save_object(file_path=self.model_training_config.Vistara_model_path,obj=vistara_model)
            logging.info(f"{'-'*30}")

            
            # AirAsia
            logging.info(f"Loading Train and Test data of AirAsia Model")
            airasia_train_data = pd.read_csv(self.data_transformation_artifact.AirAsia_train_path)
            airasia_test_data = pd.read_csv(self.data_transformation_artifact.AirAsia_test_path)

            X_train_airasia, y_train_airasia = airasia_train_data.drop(TARGET_COLUMN, axis=1), airasia_train_data[TARGET_COLUMN]
            X_test_airasia, y_test_airasia = airasia_test_data.drop(TARGET_COLUMN, axis=1), airasia_test_data[TARGET_COLUMN]

            logging.info("Training the airasia Model")
            airasia_model = self.train_model(X=X_train_airasia, y=y_train_airasia)
            
            logging.info("Predicting airasia Model")
            y_pred_airasia = airasia_model.predict(X_test_airasia)

            train_score_airasia = airasia_model.score(X_train_airasia, y_train_airasia)
            test_score_airasia = airasia_model.score(X_test_airasia, y_test_airasia)
            logging.info(f"airasia Model: Train Score={train_score_airasia} and Test Score={test_score_airasia}")
            R2_score_airasia = metrics.r2_score(y_test_airasia, y_pred_airasia)
            logging.info(f"R2_Score of airasia Model: {R2_score_airasia}")

            logging.info("Checking airasia Model is Good or Bad")
            if R2_score_airasia < self.model_training_config.expected_score:
                raise Exception(f"airasia Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}:airasia model actual score is {R2_score}")

            logging.info(f"Checking airasia Model is Overfitting or not")
            diff = abs(train_score_airasia - test_score_airasia)
            if diff > self.model_training_config.overfitting_threshold:
                raise Exception(f"airasia Train and airasia Test Score difference: {diff} is more than Overfitting threshold {self.model_training_config.overfitting_threshold}")

            logging.info(f"Saving Airasia Model Object")
            utils.save_object(file_path=self.model_training_config.AirAsia_model_path,obj=airasia_model)
            logging.info(f"{'-'*30}")

            
            # GoAir
            logging.info(f"Loading Train and Test data of GoAir Model")
            goair_train_data = pd.read_csv(self.data_transformation_artifact.GoAir_train_path)
            goair_test_data = pd.read_csv(self.data_transformation_artifact.GoAir_test_path)

            X_train_goair, y_train_goair = goair_train_data.drop(TARGET_COLUMN, axis=1), goair_train_data[TARGET_COLUMN]
            X_test_goair, y_test_goair = goair_test_data.drop(TARGET_COLUMN, axis=1), goair_test_data[TARGET_COLUMN]

            logging.info("Training the goair Model")
            goair_model = self.train_model(X=X_train_goair, y=y_train_goair)
            
            logging.info("Predicting goair Model")
            y_pred_goair = goair_model.predict(X_test_goair)

            train_score_goair = goair_model.score(X_train_goair, y_train_goair)
            test_score_goair = goair_model.score(X_test_goair, y_test_goair)
            logging.info(f"goair Model: Train Score={train_score_goair} and Test Score={test_score_goair}")
            R2_score_goair = metrics.r2_score(y_test_goair, y_pred_goair)
            logging.info(f"R2_Score of goair Model: {R2_score_goair}")

            logging.info("Checking goair Model is Good or Bad")
            if R2_score_goair < self.model_training_config.expected_score:
                raise Exception(f"goair Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}:goair model actual score is {R2_score}")

            logging.info(f"Checking goair Model is Overfitting or not")
            diff = abs(train_score_goair - test_score_goair)
            if diff > self.model_training_config.overfitting_threshold:
                raise Exception(f"goair Train and goair Test Score difference: {diff} is more than Overfitting threshold {self.model_training_config.overfitting_threshold}")

            logging.info(f"Saving GoAir Model Object")
            utils.save_object(file_path=self.model_training_config.GoAir_model_path,obj=goair_model)
            logging.info(f"{'-'*30}")


            # Prepare Artifact
            logging.info(f"Preparing Artifact")
            model_trainer_artifact = artifact_entity.ModelTrainingArtifact(
                gen_model_path=self.model_training_config.gen_model_path, 
                JetAirways_model_path=self.model_training_config.JetAirways_model_path, 
                Indigo_model_path=self.model_training_config.Indigo_model_path, 
                AirIndia_model_path=self.model_training_config.AirIndia_model_path, 
                MultipleCarriers_model_path=self.model_training_config.MultipleCarries_model_path, 
                SpiceJet_model_path=self.model_training_config.SpiceJet_model_path, 
                Vistara_model_path=self.model_training_config.Vistara_model_path, 
                AirAsia_model_path=self.model_training_config.AirAsia_model_path, 
                GoAir_model_path=self.model_training_config.GoAir_model_path
            )

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            logging.debug(str(e))
            raise FlightException(error_message=e, error_detail=sys)
