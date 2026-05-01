"""
Delta Lake advanced operations:
  - OPTIMIZE (compaction)
  - Z-ORDER
  - VACUUM
  - Time travel (read historical version)
  - Schema evolution (demonstrated in bronze)
"""

import logging

import polars as pl
from deltalake import DeltaTable, write_deltalake

from config import BRONZE_PATH, SILVER_PATH, GOLD_FEATURES_PATH

logger = logging.getLogger(__name__)


def optimize_and_zorder(path: str, zorder_cols: list[str]) -> None:
    """Run OPTIMIZE compaction followed by Z-ORDER on specified columns."""
    dt = DeltaTable(path)
    logger.info(f"OPTIMIZE compaction on {path}…")
    dt.optimize.compact()
    logger.info(f"Z-ORDER by {zorder_cols} on {path}…")
    dt.optimize.z_order(zorder_cols)
    logger.info("OPTIMIZE + Z-ORDER complete.")


def read_version(path: str, version: int) -> pl.DataFrame:
    """Time travel: read a specific version of a Delta table."""
    logger.info(f"Time travel: reading {path} @ version {version}…")
    df = pl.scan_delta(path, version=version).collect()
    logger.info(f"  → {len(df):,} rows at version {version}")
    return df


def show_history(path: str) -> None:
    """Print Delta table version history."""
    dt = DeltaTable(path)
    history = dt.history()
    for entry in history:
        logger.info(
            f"  v{entry.get('version')} | {entry.get('timestamp')} | {entry.get('operation')}"
        )


def schema_evolution_demo(path: str) -> None:
    """
    Demonstrate schema evolution: append a DataFrame with an extra column.
    deltalake write_deltalake with schema_mode='merge' will add the column.
    """
    dt = DeltaTable(path)
    current_version = dt.version()
    logger.info(f"Schema evolution demo on {path} (current version={current_version})")

    # Read a tiny slice and add a new column
    sample = pl.scan_delta(path, version=current_version).limit(10).collect()
    sample = sample.with_columns(pl.lit("schema_evolution_demo").alias("_demo_col"))
    write_deltalake(path, sample, mode="append", schema_mode="merge")
    logger.info(
        f"Schema evolution: added '_demo_col', new version={DeltaTable(path).version()}"
    )


def run_delta_utils() -> None:
    # 1. OPTIMIZE + Z-ORDER on silver
    optimize_and_zorder(SILVER_PATH, ["IATA_Code_Operating_Airline", "Origin"])

    # 2. OPTIMIZE on gold features
    optimize_and_zorder(GOLD_FEATURES_PATH, ["IATA_Code_Operating_Airline", "Origin"])

    # 3. Time travel — read bronze version 0 (first day loaded)
    v0_df = read_version(BRONZE_PATH, version=0)
    logger.info(f"Bronze v0 sample:\n{v0_df.head(3)}")

    # 4. Show version history of bronze
    show_history(BRONZE_PATH)

    # 5. Schema evolution demo on bronze
    schema_evolution_demo(BRONZE_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_delta_utils()
