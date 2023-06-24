import pymongo
import pandas as pd
import json

from flight.config import mongo_client

DATA_FILE_PATH = "/config/workspace/Data_Train.csv"
DATABASE_NAME = "flight_fare_prediction"
COLLECTION_NAME = "flight_fare"


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and Columns: {df.shape}")

    # Convert data frame to json format so that we can dump this record into mongo db
    df.reset_index(drop=True,inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    # insert converted json record to mong db
    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
