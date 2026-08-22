import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "plant_images"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = ROOT_DIR / os.getenv("MODEL_PATH", "models/plant_classifier.keras")
CLASS_NAMES_PATH = ROOT_DIR / os.getenv("CLASS_NAMES_PATH", "models/class_names.json")
TRAINING_HISTORY_PATH = MODEL_DIR / "training_history.json"

IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("EPOCHS", "20"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.15"))
VALIDATION_SIZE = float(os.getenv("VALIDATION_SIZE", "0.15"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

APP_NAME = os.getenv("APP_NAME", "Plant Identification API")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

for directory in [PROCESSED_DATA_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
