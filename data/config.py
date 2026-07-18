import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent
REPO_ROOT = DATA_DIR.parent

ZIP_PATH = Path(os.environ.get("GLANCE_ZIP_PATH", DATA_DIR / "raw" / "Glance_data.zip"))
ANN_PATH = Path(os.environ.get("GLANCE_ANN_PATH", DATA_DIR / "raw" / "glance_data.json"))

IMAGES_DIR = Path(os.environ.get("GLANCE_IMAGES_DIR", DATA_DIR / "images"))
METADATA_DIR = Path(os.environ.get("GLANCE_METADATA_DIR", DATA_DIR / "metadata"))
CATEGORIES_EXPORT_PATH = METADATA_DIR / "categories.json"
METADATA_JSON_PATH = METADATA_DIR / "metadata.json"

OUTPUT_DIR = Path(os.environ.get("GLANCE_OUTPUT_DIR", IMAGES_DIR))
INDEX_DIR = Path(os.environ.get("GLANCE_INDEX_DIR", REPO_ROOT / "outputs" / "index"))

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
FASHIONCLIP_MODEL = "patrickjohncyh/fashion-clip"
SAM_CHECKPOINT_TYPE = "vit_b"

DETECTION_BOX_THRESHOLD = 0.30
DETECTION_TEXT_THRESHOLD = 0.25

DATASET_SIZE = 1000
DATASET_RANDOM_SEED = 42
GD_SCORE_THRESHOLD = 0.3
PERSON_SCORE_THRESHOLD = 0.3
GD_CHUNK_SIZE = 16
AUTO_ANNOTATE_CHECKPOINT_EVERY = 200
CATEGORY_CONTAINMENT_THRESHOLD = 0.5

# The ONLY scenes accepted into the dataset. build_dataset.py rejects anything
# the classifier tags as runway/studio -- those two stay in the prompt
TARGET_SCENES = ["office", "street", "park", "home"]

SCENE_PROMPT_TEMPLATES = {
    "office": [
        "a photo of a person in an office interior",
        "a person standing in a corporate workplace",
        "someone working at a desk in an office",
    ],
    "street": [
        "a photo of a person on an urban street",
        "a person walking on a city sidewalk",
        "someone standing on a busy street",
    ],
    "park": [
        "a photo of a person in a park",
        "a person outdoors surrounded by trees and grass",
        "someone sitting on a park bench",
    ],
    "home": [
        "a photo of a person at home",
        "a person indoors in a living room",
        "someone relaxing in a domestic home setting",
    ],
    "runway": [
        "a photo of a model walking on a fashion runway or catwalk",
        "a runway fashion show model",
        "a model on a catwalk during a fashion show",
    ],
    "studio": [
        "a photo of a person against a plain studio backdrop",
        "a studio photoshoot with plain background",
        "a person photographed against a seamless backdrop",
    ],
}

# Single-prompt-per-scene view, only over the 4 target scenes.
SCENE_PROMPTS = {s: SCENE_PROMPT_TEMPLATES[s][0] for s in TARGET_SCENES}

SCENE_KEYWORDS = {
    "office": ["office", "workplace", "corporate", "desk", "business attire", "professional"],
    "street": ["street", "urban", "city", "sidewalk", "city walk", "downtown"],
    "park": ["park", "bench", "outdoors", "trees", "garden"],
    "home": ["home", "house", "living room", "indoors", "domestic"],
}

# Garment vocabulary -- loaded from data/metadata/categories.json, which is
# exported straight from the Fashionpedia annotations by build_dataset.py.

if not CATEGORIES_EXPORT_PATH.exists():
    raise FileNotFoundError(
        f"GARMENT_VOCAB source not found at {CATEGORIES_EXPORT_PATH}. "
        "Run data/build_dataset.py first (it exports categories.json), or "
        "set GLANCE_METADATA_DIR to point at a folder that already has one."
    )

with open(CATEGORIES_EXPORT_PATH) as _f:
    _raw_categories = json.load(_f)

_expanded = set()
for _c in _raw_categories:
    for _part in _c.split(","):
        _part = _part.strip().lower()
        if _part:
            _expanded.add(_part)


GARMENT_VOCAB = sorted(_expanded)

# Colour palette -- the full CSS3 palette via webcolors, matching what the
# dataset-build step actually used to name garment colours. 
import webcolors

COLOR_PALETTE = {name: webcolors.name_to_rgb(name) for name in webcolors.names("css3")}

STAGE1_SHORTLIST_SIZE = 50
STAGE1_WEIGHT = 0.6
RERANK_WEIGHT = 0.4
FINAL_TOP_K_DEFAULT = 5
