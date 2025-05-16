"""Functions for preprocessing the data."""
import os
from pathlib import Path
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi
from loguru import logger
import pandas as pd

from DSML.config import (
    DATASET,
    DATASET_TEST,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)


def get_raw_data(dataset:str=DATASET, dataset_test:str=DATASET_TEST)->None:
    api = KaggleApi()
    api.authenticate()

    download_folder = Path(RAW_DATA_DIR)
    zip_path = download_folder / "employee-dataset.zip"

    logger.info(f"RAW_DATA_DIR is: {RAW_DATA_DIR}")
    api.competition_download_files(dataset, path=str(download_folder))
    api.dataset_download_files(dataset_test, path=str(download_folder), unzip=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(str(download_folder))

    Path.unlink(zip_path)


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