from collections import defaultdict
from typing import Dict

import numpy as np

import data.config as config
from utils.embedder import get_embedder
from retriever.query_parser import ParsedQuery


def _garment_attrs_scores(instance_store, embedder, garment_attrs) -> Dict[str, float]:
    #Returns {image_id: score} where score = mean over pairs of the best per-pair match confidence (colour-exact-match boosted
    #else embedding similarity as fallback)."""
    if not garment_attrs:
        return {}

    per_pair_image_scores = []
    for garment, color in garment_attrs:
        query_text = f"{color + ' ' if color else ''}{garment}"
        query_vec = embedder.embed_text(query_text)

        def predicate(meta, _g=garment):
            return meta["category"] == _g

        results = instance_store.filter_search(query_vec, top_k=200, predicate=predicate)

        image_scores = defaultdict(float)
        for meta, sim in results:
            score = sim
            if color and meta.get("color") == color:
                score = min(1.0, score + 0.5)  # exact symbolic colour match bonus
            elif color and meta.get("color") not in (None, "unknown"):
                score = max(0.0, score - 0.3)  # colour present but wrong -> penalize
            image_scores[meta["image_id"]] = max(image_scores[meta["image_id"]], score)
        per_pair_image_scores.append(image_scores)

    # AND-style combination: only images scored (however weakly) across all
    # pairs get averaged; images entirely missing a pair get a 0 for it.
    all_image_ids = set()
    for d in per_pair_image_scores:
        all_image_ids |= set(d.keys())

    combined = {}
    for img_id in all_image_ids:
        vals = [d.get(img_id, 0.0) for d in per_pair_image_scores]
        combined[img_id] = float(np.mean(vals))
    return combined


def _scene_scores(global_store, scene: str) -> Dict[str, float]:
    if not scene:
        return {}
    # scene is exact-match metadata, not a vector search
    scores = {}
    for meta in global_store.metadata:
        scores[meta["image_id"]] = 1.0 if meta.get("scene") == scene else 0.0
    return scores

# how good is the vibe match? 
def _vibe_scores(global_store, embedder, vibe: str) -> Dict[str, float]:
    if not vibe:
        return {}
    query_vec = embedder.embed_text(vibe)
    results = global_store.search(query_vec, top_k=len(global_store.metadata))
    return {meta["image_id"]: sim for meta, sim in results}


def retrieve_scores(parsed: ParsedQuery, instance_store, global_store, embedder=None,
                     shortlist_size: int = config.STAGE1_SHORTLIST_SIZE):
    embedder = embedder or get_embedder()

    signal_scores = {}
    if parsed.garment_attrs:
        signal_scores["garment_attrs"] = _garment_attrs_scores(instance_store, embedder, parsed.garment_attrs)
    if parsed.scene:
        signal_scores["scene"] = _scene_scores(global_store, parsed.scene)
    if parsed.vibe:
        signal_scores["vibe"] = _vibe_scores(global_store, embedder, parsed.vibe)

    if not signal_scores:
        # nothing parsed it will fall back to treating the whole query as vibe
        signal_scores["vibe"] = _vibe_scores(global_store, embedder, parsed.raw_query)

    weight = 1.0 / len(signal_scores)
    all_image_ids = set()
    for d in signal_scores.values():
        all_image_ids |= set(d.keys())
    # make sure every image in the corpus is considered even if it scored
    # zero on every populated signal 
    all_image_ids |= {m["image_id"] for m in global_store.metadata}

    fused = []
    for img_id in all_image_ids:
        score = sum(weight * d.get(img_id, 0.0) for d in signal_scores.values())
        fused.append((img_id, score))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:shortlist_size]
