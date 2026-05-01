"""
Gold layer: two data marts.
  1. analytics  — aggregated delay stats by airport / carrier / hour / season
  2. features   — ML-ready feature table with targets
"""

import logging
import shutil
from pathlib import Path

import polars as pl
from deltalake import write_deltalake

from config import (
    SILVER_PATH,
    GOLD_ANALYTICS_PATH,
    GOLD_FEATURES_PATH,
    DELAY_THRESHOLD_MIN,
)

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "IATA_Code_Operating_Airline",
    "Origin",
    "Dest",
    "route",
    "year",
    "month",
    "day",
    "day_of_week",
    "hour",
    "season",
    "Distance",
    "DepDelay",
    "TaxiOut",
    "CRSElapsedTime",
    "ArrDelay",  # regression target
]


def build_analytics(lf: pl.LazyFrame) -> pl.DataFrame:
    return (
        lf.filter(pl.col("ArrDelay").is_not_null())
        .group_by(
            [
                "Origin",
                "IATA_Code_Operating_Airline",
                "hour",
                "season",
                "year",
                "month",
                "day",
            ]
        )
        .agg(
            [
                pl.col("ArrDelay").mean().alias("avg_arr_delay"),
                pl.col("ArrDelay").median().alias("median_arr_delay"),
                pl.col("ArrDelay").std().alias("std_arr_delay"),
                pl.len().alias("flight_count"),
                (pl.col("ArrDelay") > DELAY_THRESHOLD_MIN).mean().alias("pct_delayed"),
            ]
        )
        .sort(["year", "month", "day", "Origin", "IATA_Code_Operating_Airline"])
        .collect()
    )


def build_features(lf: pl.LazyFrame) -> pl.DataFrame:
    available = [c for c in FEATURE_COLS if c in lf.collect_schema().names()]
    return (
        lf.select(available)
        .filter(pl.col("ArrDelay").is_not_null())
        .with_columns(
            (pl.col("ArrDelay") > DELAY_THRESHOLD_MIN).cast(pl.Int8).alias("is_delayed")
        )
        .collect()
    )


def run_gold() -> None:
    for path in [GOLD_ANALYTICS_PATH, GOLD_FEATURES_PATH]:
        if Path(path).exists():
            shutil.rmtree(path)
            logger.info(f"Cleaned gold directory: {path}")

    lf = pl.scan_delta(SILVER_PATH)

    # Analytics mart
    logger.info("Building analytics mart…")
    analytics_df = build_analytics(lf)
    write_deltalake(GOLD_ANALYTICS_PATH, analytics_df, mode="overwrite")
    logger.info(f"Analytics mart: {len(analytics_df):,} rows → {GOLD_ANALYTICS_PATH}")

    # Feature table
    logger.info("Building feature table…")
    features_df = build_features(lf)
    write_deltalake(GOLD_FEATURES_PATH, features_df, mode="overwrite")
    logger.info(f"Feature table: {len(features_df):,} rows → {GOLD_FEATURES_PATH}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_gold()
