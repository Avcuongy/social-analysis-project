from pathlib import Path
import logging

from dataloader.dowloader import dowloader
from dataloader.strategy import strategy
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_COMPLETED = PROJECT_ROOT / "data" / "completed"
LOGS_DIR = PROJECT_ROOT / "logs"
TEST_DIR = PROJECT_ROOT / "test"


def _ensure_directories() -> None:
    for path in (DATA_RAW, DATA_PROCESSED, DATA_COMPLETED, LOGS_DIR, TEST_DIR):
        path.mkdir(parents=True, exist_ok=True)
        logging.info("Configuration: Initializing folder at %s", path)


def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        filemode="a",
        filename=LOGS_DIR / "config.log",
    )
    logging.info("Configuration: Starting")
    _ensure_directories()
    dowloader()
    strategy()
    logging.info("Configuration: Complete")


if __name__ == "__main__":
    main()
