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
            # index_records.json and the FAISS metadata got out of sync 
            # surface it instead of silently dropping the result
            raise KeyError(f"image_id {image_id!r} not found in {records_path}")

        image_path = Path(record["path"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # image file moved/missing since indexing 
            # query over one bad result, but don't hide it either
            image = None
            record = {**record, "load_error": str(e)}

        resolved.append((image, record, score))

    return parsed, resolved