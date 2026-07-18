import argparse
from pathlib import Path

import data.config as config
from indexer.build_index import build_index
from logging_config import setup_logging

logger = setup_logging(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--out", type=Path, default=config.INDEX_DIR)
    args = parser.parse_args()

    logger.info(f"starting indexing run: images={args.images} out={args.out}")
    try:
        build_index(args.images, args.out)
        logger.info("indexing run completed successfully")
    except Exception:
        logger.exception("indexing run failed")
        raise


if __name__ == "__main__":
    main()