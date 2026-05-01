import os
from pathlib import Path
from datetime import date, timedelta

BASE_DIR = Path(os.getenv("DATA_DIR", "data"))
RAW_DIR = BASE_DIR / "raw"
DELTA_DIR = BASE_DIR / "delta"

BRONZE_PATH = str(DELTA_DIR / "bronze" / "flights")
SILVER_PATH = str(DELTA_DIR / "silver" / "flights")
GOLD_ANALYTICS_PATH = str(DELTA_DIR / "gold" / "analytics")
GOLD_FEATURES_PATH = str(DELTA_DIR / "gold" / "features")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "flight_delay_prediction"

DELAY_THRESHOLD_MIN = 15  # FAA standard: delayed if > 15 min
ARR_DELAY_MIN = -60
ARR_DELAY_MAX = 600

# Day-level granularity for January 2024 (31 day)
DAYS = [date(2024, 1, 1) + timedelta(days=i) for i in range(31)]

# Silver merge key
MERGE_KEY_COLS = [
    "FlightDate",
    "IATA_Code_Operating_Airline",
    "Origin",
    "Dest",
    "DepTime",
    "source_day",
]
