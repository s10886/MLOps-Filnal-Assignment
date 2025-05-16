from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATASET = "employee-dataset" # original competition dataset
DATASET_TEST = "tawfikelmetwally/employee-dataset"  # test set augmented with target labels

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_NAME = "employee-attrition-prediction"  #or "employee-churn-classifier"?

categorical = [
    "Education",
    "City",
    "Gender",
    "EverBenched",
    "PaymentTier"
]

numerical = [
    "JoiningYear",
    "Age",
    "ExperienceInCurrentDomain"
]

# Zmienna docelowa
target = "LeaveOrNot"