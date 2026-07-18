import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

import data.config as config
from indexer.vector_store import VectorStore
from logging_config import setup_logging
from utils.color_extractor import extract_color
from utils.detector import get_detector
from utils.embedder import get_embedder
from utils.scene_classifier import scene_classifier
from utils.segmenter import get_segmenter

logger = setup_logging(__name__)


def build_index(images_dir: Path, out_dir: Path):
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.name != "metadata.json"
    )
    logger.info(f"found {len(image_paths)} images in {images_dir}")

    # data/build_dataset.py already classifies scene while selecting the
    # dataset and writes it to data/metadata/metadata.json so we can use that
    # Falls back to live classification for any image missing from metadata.json
    scene_from_metadata = {}
    metadata_path = config.METADATA_JSON_PATH
    if metadata_path.exists():
        with open(metadata_path) as f:
            for record in json.load(f):
                scene_from_metadata[record["file_name"]] = (
                    record["scene"], record["scene_confidence"]
                )
        logger.info(f"loaded {len(scene_from_metadata)} precomputed scene tags from {metadata_path}")

    detector = get_detector()
    segmenter = get_segmenter()
    embedder = get_embedder()
    scene_clf = scene_classifier(embedder=embedder)

    embed_dim = len(embedder.embed_text("probe"))
    instance_store = VectorStore(dim=embed_dim)
    global_store = VectorStore(dim=embed_dim)

    image_records = []
    skipped = 0

    for path in tqdm(image_paths, desc="indexing"):
        image_id = path.name
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"skipping unreadable image {image_id}: {e}")
            skipped += 1
            continue

        # garment id + colour, per detected garment
        try:
            detections = detector.detect(image)
        except Exception as e:
            logger.warning(f"detection failed for {image_id}: {e}")
            detections = []

        # detection using grounding dino followed by segmenting the detected box 
        # using SAM, then extracting the dominant color of the segmented mask
        instance_summaries = []
        for det in detections:
            try:
                mask = segmenter.segment(image, det.box)
                color = extract_color(image, mask)
                crop = image.crop(tuple(int(v) for v in det.box))
                crop_emb = embedder.embed_image(crop) if crop.size[0] > 0 and crop.size[1] > 0 else None

                meta = {
                    "image_id": image_id,
                    "category": det.label,
                    "color": color,
                    "box": det.box,
                    "det_score": det.score,
                }
                if crop_emb is not None:
                    instance_store.add(crop_emb, meta)
                instance_summaries.append({"category": det.label, "color": color})
            except Exception as e:
                logger.warning(f"instance processing failed for {image_id} ({det.label}): {e}")

        # scene (discrete) 
        if image_id in scene_from_metadata:
            scene_tag, scene_conf = scene_from_metadata[image_id]
        else:
            scene_tag, scene_conf = scene_clf.classify(image)
        logger.debug(f"{image_id}: scene={scene_tag} conf={scene_conf:.3f} instances={len(instance_summaries)}")

        # vibe/style (global embedding) 
        global_emb = embedder.embed_image(image)
        global_store.add(global_emb, {
            "image_id": image_id, "scene": scene_tag, "scene_conf": scene_conf,
        })

        image_records.append({
            "image_id": image_id, "path": str(path), "scene": scene_tag,
            "instances": instance_summaries,
        })

    instance_store.build()
    global_store.build()

    out_dir.mkdir(parents=True, exist_ok=True)
    instance_store.save(out_dir / "instance_index")
    global_store.save(out_dir / "global_index")
    with open(out_dir / "image_records.json", "w") as f:
        json.dump(image_records, f, indent=2)

    logger.info(
        f"{len(instance_store.metadata)} garment instances, "
        f"{len(global_store.metadata)} images indexed -> {out_dir} "
        f"({skipped} images skipped)"
    )