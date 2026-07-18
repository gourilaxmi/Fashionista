import argparse
from pathlib import Path

import data.config as config
from retriever import retrieve

EVAL_QUERIES = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=config.INDEX_DIR)
    parser.add_argument("--images", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    for i, query in enumerate(EVAL_QUERIES, start=1):
        parsed, results = retrieve(query, args.index, args.images, top_k=args.k)
        print(f"\n{'=' * 70}\n[{i}] {query}")
        print(f"    parsed: garment_attrs={parsed.garment_attrs} scene={parsed.scene!r} vibe={parsed.vibe!r}")
        for rank, (image_id, score) in enumerate(results, start=1):
            print(f"    {rank}. {image_id}   score={score:.4f}")


if __name__ == "__main__":
    main()
