"""
Main pipeline orchestrator: bronze → silver → gold → delta_utils → ml
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bronze import run_bronze
from silver import run_silver
from gold import run_gold
from delta_utils import run_delta_utils
from ml import run_ml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Pipeline start ===")
    run_bronze()
    run_silver()
    run_gold()
    run_delta_utils()
    run_ml()
    logger.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
