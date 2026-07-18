import json
import random
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image

import data.config as data_config
from utils.color_extractor import extract_color
from utils.detector import get_detector
from utils.embedder import get_embedder
from utils.scene_classifier import scene_classifier
from utils.segmenter import get_segmenter


#Find different categories in the annotation JSON.
def annon_vocab(ann_pth):
    with open(ann_pth) as f:
        anns = json.load(f)
    canonical_cats = {c["name"].strip().lower(): c["id"] for c in anns.get("categories", [])}
    gd_vocab = sorted(canonical_cats.keys())
    print(f"[vocab] {len(gd_vocab)} canonical categories loaded from {ann_pth}")
    return canonical_cats, gd_vocab

# Use GroundingDINO to annotate images that are missing annotations in the annotation JSON
def auto_annotate(zip_path, ann_pth, detector, canonical_cats,
                   score_threshold=data_config.GD_SCORE_THRESHOLD,
                   chunk_size=data_config.GD_CHUNK_SIZE,
                   checkpoint_every=data_config.AUTO_ANNOTATE_CHECKPOINT_EVERY):
    with open(ann_pth) as f:
        annotations = json.load(f)

    annotated_filenames = {im["file_name"] for im in annotations["images"]}
    next_image_id = max((im["id"] for im in annotations["images"]), default=0) + 1
    next_ann_id = max((a["id"] for a in annotations["annotations"]), default=0) + 1
    next_cat_id = max((c["id"] for c in annotations.get("categories", [])), default=0) + 1

    cat_name_to_id = dict(canonical_cats)
    novel_labels_seen = set()
    new_images, new_annotations, new_categories = [], [], []
    merged_path = str(Path(ann_pth).with_name("glance_data_merged.json"))

# save all annotations along with their bboxso far to a json file
    def _checkpoint_save():
        checkpoint = {
            "images": annotations["images"] + new_images,
            "annotations": annotations["annotations"] + new_annotations,
            "categories": annotations["categories"] + new_categories,
        }
        with open(merged_path, "w") as f:
            json.dump(checkpoint, f)

    with zipfile.ZipFile(zip_path) as zf:
        zip_filenames = {Path(m).name: m for m in zf.namelist()}
        unmatched = [f for f in zip_filenames if f not in annotated_filenames]
        print(f"[auto-annotate] {len(unmatched)} unmatched images found, running GroundingDINO")

        for i, fname in enumerate(unmatched):
            try:
                with zf.open(zip_filenames[fname]) as f:
                    img_bytes = f.read()
                image = Image.open(BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                print(f"[auto-annotate] skipping {fname}: {e}")
                continue

            dets = [d for d in detector.detect(image, vocab=list(canonical_cats.keys()), chunk_size=chunk_size)
                    if d.score >= score_threshold]
            if not dets:
                continue

            img_id = next_image_id
            next_image_id += 1
            new_images.append({"file_name": fname, "id": img_id, "width": image.width, "height": image.height})

            for d in dets:
                label_norm = d.label.strip().lower().rstrip(".")

                if label_norm in cat_name_to_id:
                    cat_id_to_use = cat_name_to_id[label_norm]
                else:
                    if label_norm not in novel_labels_seen:
                        novel_labels_seen.add(label_norm)
                        print(f"[auto-annotate] novel label not in JSON taxonomy: '{label_norm}'")
                    cat_name_to_id[label_norm] = next_cat_id
                    new_categories.append({"id": next_cat_id, "name": label_norm})
                    cat_id_to_use = next_cat_id
                    next_cat_id += 1

                x0, y0, x1, y1 = d.box
                new_annotations.append({
                    "id": next_ann_id,
                    "image_id": img_id,
                    "category_id": cat_id_to_use,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                })
                next_ann_id += 1

            if (i + 1) % 100 == 0:
                print(f"[auto-annotate] {i+1}/{len(unmatched)} scanned, {len(new_images)} kept")
            if (i + 1) % checkpoint_every == 0:
                _checkpoint_save()
                print(f"[auto-annotate] checkpoint saved at {i+1}/{len(unmatched)} -> {merged_path}")

    annotations["images"].extend(new_images)
    annotations["annotations"].extend(new_annotations)
    annotations["categories"].extend(new_categories)

    with open(merged_path, "w") as f:
        json.dump(annotations, f)

    print(f"[auto-annotate] done: {len(annotations['images'])} images, "
          f"{len(annotations['categories'])} categories ({len(novel_labels_seen)} novel labels minted)")
    return merged_path

# find category, color pairs for each image, and filter by primary person. 

def category_dict(annotations):
    imgs = {im["file_name"]: im for im in annotations["images"]}
    cat_id = {c["id"]: c["name"] for c in annotations.get("categories", [])}
    anns = defaultdict(list)
    for ann in annotations["annotations"]:
        anns[ann["image_id"]].append(ann)
    return imgs, cat_id, anns

# check if each category among background has sufficient images
def category_check(keys, counts, n, denom=20):
    return any(counts[k] < (n / denom) for k in keys) or not keys

# This is important for images with multiple people, where we want to ensure that the garment attributes are correctly 
# associated with the primary subject.
def detect_primary_person(image, detector, score_threshold=data_config.PERSON_SCORE_THRESHOLD):
    dets = [d for d in detector.detect(image, vocab=["person"], chunk_size=1) if d.score >= score_threshold]
    if not dets:
        return None

    def area(box):
        x0, y0, x1, y1 = box
        return max(0, x1 - x0) * max(0, y1 - y0)

    primary = max(dets, key=lambda d: area(d.box))
    return primary.box


def bbox_contained_ratio(garment_box, person_box):
    gx0, gy0, gx1, gy1 = garment_box
    px0, py0, px1, py1 = person_box
    ix0, iy0 = max(gx0, px0), max(gy0, py0)
    ix1, iy1 = min(gx1, px1), min(gy1, py1)
    inter_area = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    garment_area = max(0, gx1 - gx0) * max(0, gy1 - gy0)
    if garment_area == 0:
        return 0.0
    return inter_area / garment_area

# returns a list of (category, color) pairs for each image.
def instance_attributes(image, anns, segmenter, cat_id, primary_box=None,
                         containment_threshold=data_config.CATEGORY_CONTAINMENT_THRESHOLD):
    w_img, h_img = image.size
    pairs = []
    for ann in anns:
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = bbox
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(w_img, int(x + w)), min(h_img, int(y + h))
        if x1 <= x0 or y1 <= y0:
            continue

        if primary_box is not None:
            ratio = bbox_contained_ratio((x0, y0, x1, y1), primary_box)
            if ratio < containment_threshold:
                continue

        try:
            mask = segmenter.segment(image, (x0, y0, x1, y1))
            color = extract_color(image, mask)
        except Exception as e:
            print(f"segmentation/colour extraction failed for bbox {bbox}: {e}")
            continue
        category_name = cat_id.get(ann["category_id"], str(ann["category_id"]))
        pairs.append({"category": category_name, "color": color})
    return pairs

# Build a dataset of images and metadata from the raw zip and annotation JSON.
# this json will also contain the attributes of auto annotated images.
def build_dataset(zip_path=data_config.ZIP_PATH, ann_pth=data_config.ANN_PATH,
                   images=data_config.IMAGES_DIR, metadata=data_config.METADATA_DIR,
                   n=data_config.DATASET_SIZE, seed=data_config.DATASET_RANDOM_SEED,
                   use_grounding_dino=True):
    zip_path, ann_pth_p, images, metadata = Path(zip_path), Path(ann_pth), Path(images), Path(metadata)

    if not zip_path.exists():
        raise FileNotFoundError(f"Image zip not found: {zip_path}")
    if not ann_pth_p.exists():
        raise FileNotFoundError(f"Annotation JSON not found: {ann_pth_p}")

    images.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)

    canonical_cats, gd_vocab = annon_vocab(ann_pth_p)
    detector = get_detector()

    if use_grounding_dino:
        ann_pth_p = Path(auto_annotate(zip_path, ann_pth_p, detector, canonical_cats))

    print(f"loading annotations from {ann_pth_p}")
    with open(ann_pth_p) as f:
        annotations = json.load(f)
    print(f"{len(annotations['images'])} total images, {len(annotations.get('categories', []))} categories")

    imgs, cat_id, anns = category_dict(annotations)

    embedder = get_embedder()
    scene_clf = scene_classifier(embedder=embedder)
    seg = get_segmenter()

    # how many images per scene
    # it is also important as many images are from runway/studio, which we want to avoid. 
    # Some runway was classified as "indoor" by the scene classifier or as street for different scenes.
    # so i asked then to identify runway and then discard it
    scenes = data_config.TARGET_SCENES
    env_quota = n // len(scenes)
    env_counts = {s: 0 for s in scenes}
    print(f"environment : {env_quota} images per scene ({scenes})")

    category_counts = defaultdict(int)
    color_counts = defaultdict(int)
    metadata_records = []
    random.seed(seed)
    no_primary_subject_skipped = 0
    rejected_scene_counts = defaultdict(int)

    print(f"opening zip {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        all_members = [m for m in zf.namelist() if Path(m).name in imgs]
        random.shuffle(all_members)
        print(f"{len(all_members)} zip entries match annotated images")

        for member in all_members:
            if len(metadata_records) >= n:
                break

            fname = Path(member).name
            img_meta = imgs[fname]
            anns_here = anns.get(img_meta["id"], [])
            cats_here = {a["category_id"] for a in anns_here}

            category_ok = category_check(cats_here, category_counts, n) or len(metadata_records) < n * 0.6
            if not category_ok:
                continue

            try:
                with zf.open(member) as f:
                    img_bytes = f.read()
                image = Image.open(BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                print(f"failed to open {fname}: {e}")
                continue

            scene_tag, scene_conf = scene_clf.classify(image)
            if scene_tag not in scenes:
                rejected_scene_counts[scene_tag] += 1
                continue
            if env_counts[scene_tag] >= env_quota:
                continue

            primary_box = detect_primary_person(image, detector)
            if primary_box is None:
                no_primary_subject_skipped += 1
                instance_pairs = instance_attributes(image, anns_here, seg, cat_id)
            else:
                instance_pairs = instance_attributes(image, anns_here, seg, cat_id, primary_box=primary_box)

            colors_here = {p["color"] for p in instance_pairs}
            color_ok = category_check(colors_here, color_counts, n) or len(metadata_records) < n * 0.6
            if not color_ok:
                continue

            image.save(images / fname)
            env_counts[scene_tag] += 1
            for c in cats_here:
                category_counts[c] += 1
            for col in colors_here:
                color_counts[col] += 1

            categories_here = sorted(cat_id.get(c, str(c)) for c in cats_here)
            metadata_records.append({
                "file_name": fname,
                "fashionpedia_image_id": img_meta["id"],
                "attributes": instance_pairs,
                "categories": categories_here,
                "colors": sorted(colors_here),
                "scene": scene_tag,
                "scene_confidence": round(scene_conf, 4),
            })

            if len(metadata_records) % 50 == 0:
                print(f"progress: {len(metadata_records)}/{n} images selected  env_counts={env_counts}")

    if len(metadata_records) < n:
        print(f"only found {len(metadata_records)}/{n} images meeting criteria")
        for s in scenes:
            if env_counts[s] < env_quota:
                print(f"  scene '{s}' underfilled: {env_counts[s]}/{env_quota}")
    print(f"rejected as runway/studio: {dict(rejected_scene_counts)}")
    print(f"{no_primary_subject_skipped} images had no confident person detection (used unfiltered attributes)")

    with open(metadata / "metadata.json", "w") as f:
        json.dump(metadata_records, f, indent=2)

    with open(data_config.CATEGORIES_EXPORT_PATH, "w") as f:
        json.dump(gd_vocab, f, indent=2)
    print(f"exported {len(gd_vocab)} categories -> {data_config.CATEGORIES_EXPORT_PATH}")

    print(f"done: {len(metadata_records)} images -> {images}")
    print(f"metadata written to {metadata / 'metadata.json'}")
    return images