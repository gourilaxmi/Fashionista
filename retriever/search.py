from collections import defaultdict
from typing import Dict

import numpy as np

import data.config as config
from utils.embedder import get_embedder
from retriever.query_parser import ParsedQuery


def _garment_attrs_scores(instance_store, embedder, garment_attrs) -> Dict[str, float]:
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
                score = min(1.0, score + 0.5)
            elif color and meta.get("color") not in (None, "unknown"):
                score = max(0.0, score - 0.3)
            image_scores[meta["image_id"]] = max(image_scores[meta["image_id"]], score)
        per_pair_image_scores.append(image_scores)

    all_image_ids = set()
    for d in per_pair_image_scores:
        all_image_ids |= set(d.keys())

    combined = {}
    for img_id in all_image_ids:
        # only average over pairs that had a candidate match for this image;
        # missing pairs are excluded rather than forced to 0, so partial
        # multi-attribute matches aren't crushed as harshly as pure AND.
        vals = [d[img_id] for d in per_pair_image_scores if img_id in d]
        combined[img_id] = float(np.mean(vals)) if vals else 0.0
    return combined


def _scene_scores(global_store, scene: str) -> Dict[str, float]:
    if not scene:
        return {}
    scores = {}
    for meta in global_store.metadata:
        scores[meta["image_id"]] = 1.0 if meta.get("scene") == scene else 0.0
    return scores


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
        signal_scores["vibe"] = _vibe_scores(global_store, embedder, parsed.raw_query)

    weight = 1.0 / len(signal_scores)
    all_image_ids = set()
    for d in signal_scores.values():
        all_image_ids |= set(d.keys())
    all_image_ids |= {m["image_id"] for m in global_store.metadata}

    fused = []
    for img_id in all_image_ids:
        score = sum(weight * d.get(img_id, 0.0) for d in signal_scores.values())
        fused.append((img_id, score))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:shortlist_size]