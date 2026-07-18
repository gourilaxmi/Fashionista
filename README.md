# Multimodal Fashion & Context Retrieval

A compositional, attribute-aware image retrieval system built for the Glance ML
Internship assignment. Given a natural language query like *"A red tie and a
white shirt in a formal setting"*, the system returns the top-k matching images
from a fashion image database.

## Why not just CLIP?

Vanilla CLIP encodes a whole image as a single vector. That means "red shirt,
blue pants" and "blue shirt, red pants" can land close together in embedding
space, because nothing forces colour to bind to a specific garment. This
project fixes that by detecting garments individually, extracting colour
per-region, and keeping scene/vibe as separate, explicit signals instead of
folding everything into one blurry global embedding.

## Architecture: three signals, fused adaptively

| Signal | What it captures | How | Used for |
|---|---|---|---|
| **1. Garment + Colour** | *which garment, what colour, where* | Grounding DINO (open-vocab garment detection) -> SAM (pixel-precise mask) -> dominant colour from masked pixels + FashionCLIP crop embedding | "red tie", "yellow raincoat" |
| **2. Scene** | *where in the world* | FashionCLIP zero-shot classification against a fixed scene-prompt list (office / park / street / home) | "in a modern office" |
| **3. Vibe / Style** | *the overall look, no fixed vocabulary* | FashionCLIP global image embedding, matched by cosine similarity | "casual weekend outfit" |

At query time, the query is parsed into slots (garment+colour pairs / scene /
vibe). Only populated slots contribute to the fused score, so a
colour-only query doesn't get diluted by an empty scene signal.

Retrieval is currently single-stage: the fused score above ranks the whole
corpus directly. A stage-2 re-verification pass (re-running Grounding DINO
against only the shortlist to double-check compositional binding on
ambiguous queries) is a natural next step and is called out under Future Work
-- it is not implemented in this submission, so results rely entirely on the
garment+colour signal above to keep "red tie, white shirt" separated from its
reverse.

Full design rationale, alternatives considered, and tradeoffs are in the
submission PDF write-up.

## Repo layout

```
glance-fashion-retrieval/
├── data/
│   ├── config.py               # the ONE config file -- every module imports this
│   ├── build_dataset.py        # subsamples the Fashionpedia zip+annotations
│   │                            # into a diverse 500-1000 image working set
│   │                            # (category/colour/scene quotas)
│   ├── run_dataset.py          # CLI entrypoint: python -m data.run_dataset
│   ├── images/                 # dataset images (git-ignored, see below)
│   └── metadata/
│       ├── metadata.json       # per-image garment/colour/scene records
│       └── categories.json     # exported Fashionpedia category vocab
│                                # (this is what config.py's GARMENT_VOCAB loads)
├── utils/
│   ├── detector.py             # Grounding DINO wrapper (open-vocab detection)
│   ├── segmenter.py            # SAM wrapper (box -> mask)
│   ├── embedder.py             # FashionCLIP wrapper (image/text embeddings)
│   ├── color_extractor.py      # dominant colour from masked pixels (CSS3 palette)
│   └── scene_classifier.py     # zero-shot scene tagging
├── indexer/
│   ├── vector_store.py         # FAISS wrapper (instance_index + global_index)
│   └── build_index.py          # orchestrates the full indexing pipeline
├── retriever/
│   ├── query_parser.py         # query -> {garment_attrs, scene, vibe} slots
│   └── search.py                # stage-1 retrieval + adaptive fusion
├── scripts/
│   ├── run_indexing.py         # CLI entrypoint: build the index
│   └── run_query.py            # CLI entrypoint: run a single query
├── eval/
│   └── eval_queries.py         # runs the 5 assignment eval queries
├── requirements.txt
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Build the working dataset (500-1000 images). By default this reads/writes
#    under data/ (data/raw, data/images, data/metadata). To point at a
#    different location instead, set env vars BEFORE running -- see
#    data/config.py for the full list (GLANCE_ZIP_PATH, GLANCE_IMAGES_DIR, etc.)
python -m data.run_dataset

# 2. Build the index (runs DINO + SAM + FashionCLIP over every image)
python -m scripts.run_indexing --images data/images --out outputs/index

# 3. Query
python -m scripts.run_query --index outputs/index \
  --query "A red tie and a white shirt in a formal setting" --k 5

# 4. Run the assignment's 5 evaluation queries end-to-end
python -m eval.eval_queries --index outputs/index --images data/images
```

## Dataset source

`data/build_dataset.py` reads `config.ZIP_PATH` (the Fashionpedia image zip)
and `config.ANN_PATH` (its annotation file), then samples a diverse subset
across garment categories, colours, and the 4 target scenes (office, street,
park, home) using the accompanying annotations. Images the scene classifier
tags as runway/studio are rejected outright -- those two categories exist
only as distractor classes so the classifier has somewhere to put editorial
shots instead of misfiling them as office/home. Fashionpedia annotations are
used only to select a *diverse subset* and to enforce category/colour/scene
quotas -- never as ground-truth labels fed into the retrieval pipeline itself
(that would defeat the zero-shot claim).

`data/config.py`'s `GARMENT_VOCAB` and `COLOR_PALETTE` are not hardcoded:
`GARMENT_VOCAB` is loaded from `data/metadata/categories.json` (exported by
`build_dataset.py`), and `COLOR_PALETTE` is the full CSS3 palette via
`webcolors`, matching what actually generated the dataset's colour labels.

## Scalability notes

- `IndexFlatIP` is used by default for small subsets; switch to
  `IndexIVFPQ` in `vector_store.py` (flag already present) for corpora
  approaching 1M images -- sublinear search, compressed vectors.
- Instance vectors are shardable by garment category, shrinking the ANN
  search space before scoring.
- Scene tags are stored as flat metadata (not vector search), so
  scene-filtering is O(1) regardless of corpus size.

## Future work

- **Locations/weather**: add a place-tagging signal similar to the scene
  classifier (city/landmark zero-shot prompts, or a geotag if available) and
  a weather signal (rain/snow/sun) using the same fused-scoring pattern as
  scene -- another populated slot, weighted in only when the query mentions it.
- **Precision**: implement the stage-2 re-verification pass described above
  (re-run Grounding DINO on the literal query phrase against only the
  shortlisted candidates) to catch cases where the fused score ranks a
  near-miss above the true compositional match.
