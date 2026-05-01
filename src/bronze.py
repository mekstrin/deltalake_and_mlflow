"""
Bronze layer: load raw CSV files into Delta table, one batch per day (append mode).
This creates a real version history and simulates incremental data arrival.
"""

import logging
import shutil
from datetime import date
from pathlib import Path
from datetime import datetime, timezone

import polars as pl
from deltalake import write_deltalake

from config import RAW_DIR, BRONZE_PATH, DAYS

logger = logging.getLogger(__name__)

BRONZE_SCHEMA = {
    "FlightDate": pl.Utf8,
    "IATA_Code_Operating_Airline": pl.Utf8,
    "Flight_Number_Operating_Airline": pl.Int32,
    "Origin": pl.Utf8,
    "Dest": pl.Utf8,
    "CRSDepTime": pl.Int32,
    "DepTime": pl.Float64,
    "DepDelay": pl.Float64,
    "TaxiOut": pl.Float64,
    "WheelsOff": pl.Float64,
    "WheelsOn": pl.Float64,
    "TaxiIn": pl.Float64,
    "CRSArrTime": pl.Int32,
    "ArrTime": pl.Float64,
    "ArrDelay": pl.Float64,
    "Cancelled": pl.Float64,
    "CancellationCode": pl.Utf8,
    "Diverted": pl.Float64,
    "CRSElapsedTime": pl.Float64,
    "ActualElapsedTime": pl.Float64,
    "AirTime": pl.Float64,
    "Distance": pl.Float64,
    "CarrierDelay": pl.Float64,
    "WeatherDelay": pl.Float64,
    "NASDelay": pl.Float64,
    "SecurityDelay": pl.Float64,
    "LateAircraftDelay": pl.Float64,
}


def load_day(day: date) -> pl.DataFrame:
    combined_path = RAW_DIR / "flight_data_2018_2024.csv"

    if combined_path.exists():
        df = pl.read_csv(
            combined_path,
            schema_overrides=BRONZE_SCHEMA,
            null_values=["", "NA", "N/A"],
            ignore_errors=True,
            truncate_ragged_lines=True,
        )
    else:
        raise FileNotFoundError(f"No data found: {combined_path}")

    df = df.with_columns(
        pl.col("FlightDate")
        .str.to_date(format="%Y-%m-%d", strict=False)
        .alias("_parsed")
    )
    df = df.filter(pl.col("_parsed") == day)
    df = df.drop("_parsed")

    df = df.with_columns(
        [
            pl.lit(day.isoformat()).alias("source_day"),
            pl.lit(day.month).cast(pl.Int32).alias("source_month"),
            pl.lit(datetime.now(timezone.utc).isoformat()).alias("load_ts"),
        ]
    )

    logger.info(f"Loaded day {day}: {len(df):,} rows, {len(df.columns)} cols")
    return df


def run_bronze(days: list[date] = None) -> None:
    """Load each day as a separate append batch into the bronze Delta table."""
    if days is None:
        days = DAYS

    if Path(BRONZE_PATH).exists():
        shutil.rmtree(BRONZE_PATH)
        logger.info(f"Cleaned bronze directory: {BRONZE_PATH}")

    for day in days:
        try:
            df = load_day(day)
        except FileNotFoundError as e:
            logger.warning(str(e))
            continue

        write_deltalake(BRONZE_PATH, df, mode="append", schema_mode="merge")
        logger.info(f"Bronze: wrote day {day} (append) → {BRONZE_PATH}")

    logger.info("Bronze layer complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_bronze()
