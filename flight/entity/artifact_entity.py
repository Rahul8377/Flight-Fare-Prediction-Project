from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_file_path:str
    train_file_path:str
    test_file_path:str


@dataclass
class DataValidationArtifact:
    report_file_path:str


@dataclass
class DataTransformationArtifact:
    transform_train_path:str
    transform_test_path:str
    JetAirways_train_path:str
    JetAirways_test_path:str
    Indigo_train_path:str
    Indigo_test_path:str
    AirIndia_train_path:str
    AirIndia_test_path:str
    MultipleCarriers_train_path:str
    MultipleCarriers_test_path:str
    SpiceJet_train_path:str
    SpiceJet_test_path:str
    Vistara_train_path:str
    Vistara_test_path:str
    AirAsia_train_path:str
    AirAsia_test_path:str
    GoAir_train_path:str
    GoAir_test_path:str

@dataclass
class ModelTrainingArtifact:
    gen_model_path:str
    JetAirways_model_path:str
    Indigo_model_path:str
    AirIndia_model_path:str
    MultipleCarriers_model_path:str
    SpiceJet_model_path:str
    Vistara_model_path:str
    AirAsia_model_path:str
    GoAir_model_path:str

class ModelEvaluationArtifact:...

@dataclass
class ModelPusherArtifact:
    saved_model_dir:str