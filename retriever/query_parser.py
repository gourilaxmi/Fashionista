import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import data.config as config


@dataclass
class ParsedQuery:
    garment_attrs: List[Tuple[str, Optional[str]]] = field(default_factory=list)
    scene: Optional[str] = None
    vibe: Optional[str] = None
    raw_query: str = ""


_GARMENT_SORTED = sorted(config.GARMENT_VOCAB, key=len, reverse=True)
_COLOR_SORTED = sorted(config.COLOR_PALETTE.keys(), key=len, reverse=True)

assert set(config.SCENE_KEYWORDS) == set(config.SCENE_PROMPTS), (
    "config.SCENE_KEYWORDS and config.SCENE_PROMPTS must define the same "
    "scene tags -- update both together."
)
_SCENE_KEYWORDS = config.SCENE_KEYWORDS


def _find_garment_color_pairs(text: str) -> List[Tuple[str, Optional[str]]]:
    pairs = []
    lowered = text.lower()
    for garment in _GARMENT_SORTED:
        for m in re.finditer(rf"\b{re.escape(garment)}\b", lowered):
            window_start = max(0, m.start() - 25)
            window = lowered[window_start:m.start()]
            color = next((c for c in _COLOR_SORTED if re.search(rf"\b{c}\b", window)), None)
            pairs.append((garment, color))
    seen = set()
    unique = []
    for g, c in pairs:
        key = (g, c)
        if key not in seen:
            seen.add(key)
            unique.append((g, c))
    return unique


def _find_scene(text: str) -> Optional[str]:
    lowered = text.lower()
    for scene, keywords in _SCENE_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return scene
    return None


def parse_query(query: str) -> ParsedQuery:
    garment_attrs = _find_garment_color_pairs(query)
    scene = _find_scene(query)

    residual = query
    for garment, color in garment_attrs:
        residual = re.sub(rf"\b{re.escape(garment)}\b", "", residual, flags=re.I)
        if color:
            residual = re.sub(rf"\b{re.escape(color)}\b", "", residual, flags=re.I)

    if scene:
        for kw in config.SCENE_KEYWORDS[scene]:
            residual = re.sub(rf"\b{re.escape(kw)}\b", "", residual, flags=re.I)

    residual = re.sub(r"\s+", " ", residual).strip(" .,")

    vibe = residual if len(residual.split()) >= 1 else None

    return ParsedQuery(garment_attrs=garment_attrs, scene=scene, vibe=vibe, raw_query=query)