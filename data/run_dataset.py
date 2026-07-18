from logging_config import setup_logging
from data.build_dataset import build_dataset

logger = setup_logging(__name__)


def main():
    logger.info("starting dataset build")
    try:
        build_dataset()
        logger.info("dataset build completed successfully")
    except Exception:
        logger.exception("dataset build failed")
        raise


if __name__ == "__main__":
    main()