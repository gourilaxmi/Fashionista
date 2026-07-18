import argparse
from pathlib import Path

import data.config as config
from logging_config import setup_logging
from retriever import retrieve

logger = setup_logging(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=config.INDEX_DIR)
    parser.add_argument("--images", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--k", type=int, default=config.FINAL_TOP_K_DEFAULT)
    args = parser.parse_args()

    logger.info(f"query received: {args.query!r}")
    try:
        parsed, results = retrieve(args.query, args.index, args.images, top_k=args.k)
    except Exception:
        logger.exception(f"query failed: {args.query!r}")
        raise

    logger.info(
        f"parsed slots: garment_attrs={parsed.garment_attrs} "
        f"scene={parsed.scene!r} vibe={parsed.vibe!r}"
    )
    logger.info(f"returned {len(results)} results")

    print(f"\nQuery: {args.query!r}")
    print(f"Parsed slots: garment_attrs={parsed.garment_attrs} scene={parsed.scene!r} vibe={parsed.vibe!r}\n")
    print(f"Top {len(results)} results:")
    for rank, (image_id, score) in enumerate(results, start=1):
        print(f"  {rank}. {image_id}   score={score:.4f}")
        logger.info(f"  rank={rank} image_id={image_id} score={score:.4f}")


if __name__ == "__main__":
    main()