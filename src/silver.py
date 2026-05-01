"""
Silver layer: clean, enrich, and deduplicate flight data.
Writes with partition_by=["year","month","day"].
Re-runs use MERGE to update existing records without duplication.
"""

import logging
import shutil
from pathlib import Path

import polars as pl
from deltalake import DeltaTable, write_deltalake

from config import (
    BRONZE_PATH,
    SILVER_PATH,
    ARR_DELAY_MIN,
    ARR_DELAY_MAX,
    MERGE_KEY_COLS,
)

logger = logging.getLogger(__name__)


def _season(month_col: pl.Expr) -> pl.Expr:
    return (
        pl.when(month_col.is_in([12, 1, 2]))
        .then(pl.lit("winter"))
        .when(month_col.is_in([3, 4, 5]))
        .then(pl.lit("spring"))
        .when(month_col.is_in([6, 7, 8]))
        .then(pl.lit("summer"))
        .otherwise(pl.lit("fall"))
        .alias("season")
    )


def transform(df: pl.DataFrame) -> pl.DataFrame:
    # 1. Filter cancelled flights
    df = df.filter(pl.col("Cancelled") != 1.0)

    # 2. Drop rows with nulls in critical columns
    df = df.drop_nulls(subset=["ArrDelay", "DepTime", "Origin", "Dest", "FlightDate"])

    # 3. Remove outliers
    df = df.filter(
        (pl.col("ArrDelay") >= ARR_DELAY_MIN) & (pl.col("ArrDelay") <= ARR_DELAY_MAX)
    )

    # 4. Normalize text columns
    df = df.with_columns(
        [
            pl.col("IATA_Code_Operating_Airline").str.strip_chars().str.to_uppercase(),
            pl.col("Origin").str.strip_chars().str.to_uppercase(),
            pl.col("Dest").str.strip_chars().str.to_uppercase(),
        ]
    )

    # 5. Parse FlightDate and derive temporal features
    df = df.with_columns(
        pl.col("FlightDate")
        .str.to_date(format="%Y-%m-%d", strict=False)
        .alias("FlightDate_parsed")
    )
    df = df.with_columns(
        [
            pl.col("FlightDate_parsed").dt.year().cast(pl.Int32).alias("year"),
            pl.col("FlightDate_parsed").dt.month().cast(pl.Int32).alias("month"),
            pl.col("FlightDate_parsed").dt.day().cast(pl.Int32).alias("day"),
            pl.col("FlightDate_parsed")
            .dt.weekday()
            .cast(pl.Int32)
            .alias("day_of_week"),
        ]
    )

    # 6. Hour from DepTime (HHMM format)
    df = df.with_columns((pl.col("DepTime").cast(pl.Int32) // 100).alias("hour"))

    # 7. Season
    df = df.with_columns(_season(pl.col("month")))

    # 8. Route
    df = df.with_columns(
        (pl.col("Origin") + pl.lit("-") + pl.col("Dest")).alias("route")
    )

    # 9. Drop parse helper
    df = df.drop("FlightDate_parsed")

    return df


def run_silver() -> None:
    if Path(SILVER_PATH).exists():
        shutil.rmtree(SILVER_PATH)
        logger.info(f"Cleaned silver directory: {SILVER_PATH}")

    logger.info("Reading bronze table (lazy)…")
    lf = pl.scan_delta(BRONZE_PATH)

    # Show query plan with pushdowns
    plan = lf.explain(optimized=True)
    logger.info(f"Bronze scan plan:\n{plan}")

    df = lf.collect()
    logger.info(f"Bronze rows: {len(df):,}")

    df = transform(df)
    logger.info(f"Silver rows after cleaning: {len(df):,}")

    # Check if silver table already exists → use MERGE, else write fresh
    try:
        dt = DeltaTable(SILVER_PATH)
        _merge_silver(dt, df)
    except Exception:
        # First run — write fresh
        write_deltalake(
            SILVER_PATH,
            df,
            mode="overwrite",
            partition_by=["year", "month", "day"],
        )
        logger.info(f"Silver: initial write complete → {SILVER_PATH}")


def _merge_silver(dt: DeltaTable, df: pl.DataFrame) -> None:
    """MERGE new/updated records into the silver table by flight key."""
    predicate = " AND ".join(f"s.{c} = t.{c}" for c in MERGE_KEY_COLS)
    (
        dt.merge(
            source=df,
            predicate=f"{predicate}",
            source_alias="s",
            target_alias="t",
        )
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute()
    )
    logger.info(f"Silver: MERGE complete → {SILVER_PATH}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_silver()
