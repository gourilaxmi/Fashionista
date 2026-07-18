import json
from pathlib import Path

from PIL import Image

import data.config as config
from indexer.vector_store import VectorStore
from utils.embedder import get_embedder
from retriever.query_parser import parse_query
from retriever.search import retrieve_scores


def retrieve(query: str, index_dir: Path, images_dir: Path = None, top_k: int = config.FINAL_TOP_K_DEFAULT):
    index_dir = Path(index_dir)
    images_dir = Path(images_dir or config.OUTPUT_DIR)

    instance_store = VectorStore.load(index_dir / "instance_index")
    global_store = VectorStore.load(index_dir / "global_index")
    embedder = get_embedder()

    parsed = parse_query(query)
    results = retrieve_scores(parsed, instance_store, global_store, embedder=embedder)

    return parsed, results[:top_k]


def retrieve_with_images(query: str, index_dir: Path, images_dir: Path = None,
                          top_k: int = config.FINAL_TOP_K_DEFAULT):

    index_dir = Path(index_dir)

    parsed, results = retrieve(query, index_dir, images_dir, top_k)

    records_path = index_dir / "image_records.json"
    with open(records_path) as f:
        records_by_id = {r["image_id"]: r for r in json.load(f)}

    resolved = []
    for image_id, score in results:
        record = records_by_id.get(image_id)
        if record is None:
            raise KeyError(f"image_id {image_id!r} not found in {records_path}")

        image_path = Path(record["path"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            image = None
            record = {**record, "load_error": str(e)}

        resolved.append((image, record, score))

    return parsed, resolved


def save_results_plot(query: str, resolved: list, output_dir: Path = Path("results"), filename: str = None):
    # save the images
    import numpy as np
    import re
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend, safe for scripts/headless runs
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = max(len(resolved), 1)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (image, record, score) in zip(axes, resolved):
        if image is None:
            ax.text(0.5, 0.5, "load error", ha="center", va="center")
        else:
            ax.imshow(image)
        ax.set_title(f"{score:.3f}", fontsize=10)
        ax.axis("off")

    fig.suptitle(query, fontsize=11)
    plt.tight_layout()

    if filename is None:
        safe_query = re.sub(r"[^a-zA-Z0-9]+", "_", query.strip().lower())[:50]
        filename = f"{safe_query}.png"

    out_path = output_dir / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path