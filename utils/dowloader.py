import kagglehub
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
LOG_FILE = PROJECT_ROOT / "logs" / "data.log"


def dowloader():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        filemode="a",
        filename=LOG_FILE,
    )
    logger = logging.getLogger(__name__)
    try:
        logger.info("Downloading dataset...")
        logger.info(f"Data dir: {DATA_DIR}")

        try:
            path = kagglehub.dataset_download(
                "kausthubkannan/github-social-network",
                output_dir=str(DATA_DIR),
                force_download=True,
            )
            logger.info(f"kagglehub returned: {path}")
        except Exception:
            logger.exception("kagglehub download failed")
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    dowloader()
