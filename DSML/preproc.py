"""Functions for preprocessing the data."""
import os
from pathlib import Path
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi
from loguru import logger
import pandas as pd

from DSML.config import (
    DATASET,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)


def get_raw_data(dataset:str=DATASET)->None:
    api = KaggleApi()
    api.authenticate()

    logger.info(f"RAW_DATA_DIR is: {RAW_DATA_DIR}")
    api.dataset_download_files(
        dataset="tawfikelmetwally/employee-dataset",
        path=str(RAW_DATA_DIR),
        unzip=True
    )



def preprocess_df(file:str|Path)->str|Path:
    """Preprocess datasets."""
    _, file_name = os.path.split(file)
    df_data = pd.read_csv(file)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    outfile_path = PROCESSED_DATA_DIR / file_name
    df_data.to_csv(outfile_path, index=False)

    return outfile_path


if __name__=="__main__":
    # get the train and test sets from default location
    logger.info("getting datasets")
    get_raw_data()

    # preprocess both sets
    logger.info("preprocessing train.csv")
    preprocess_df(RAW_DATA_DIR / "train.csv")
    logger.info("preprocessing test.csv")
    preprocess_df(RAW_DATA_DIR / "test.csv")