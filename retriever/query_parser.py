import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import data.config as config

try:
    from rapidfuzz import process, fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    import difflib
    _HAS_RAPIDFUZZ = False


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

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "for", "with",
    "of", "to", "is", "are", "was", "were", "be", "being", "been", "this",
    "that", "these", "those", "it", "its", "as", "by", "from", "into",
    "someone", "person", "wearing", "wears", "wear",
}

_FUZZY_THRESHOLD = 82  # 0-100 similarity


def _normalize_plural(token: str) -> str:
    # Convert plural -> singular
    if token in config.GARMENT_VOCAB:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("shes", "ches", "sses", "xes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and token[:-1] in config.GARMENT_VOCAB:
        return token[:-1]
    return token


def _fuzzy_match(token: str, vocab: List[str]) -> Optional[str]:
    # Return the closest vocab term if within threshold, else None.
    if not token or len(token) < 3:
        return None
    if _HAS_RAPIDFUZZ:
        match = process.extractOne(token, vocab, scorer=fuzz.ratio)
        if match and match[1] >= _FUZZY_THRESHOLD:
            return match[0]
        return None
    else:
        close = difflib.get_close_matches(token, vocab, n=1, cutoff=_FUZZY_THRESHOLD / 100)
        return close[0] if close else None


def _canonicalize_tokens(text: str) -> str:
    """Normalize plurals + typo-tolerant fuzzy match, word by word,
    for garment and color vocabulary only. Leaves everything else untouched."""
    words = re.findall(r"[a-zA-Z\-']+|\s+|[^\sa-zA-Z\-']", text)
    out = []
    for w in words:
        lw = w.lower()
        if not lw.strip() or not re.match(r"^[a-zA-Z\-']+$", lw):
            out.append(w)
            continue

        canon = _normalize_plural(lw)

        if canon not in config.GARMENT_VOCAB and canon not in config.COLOR_PALETTE:
            fuzzy_g = _fuzzy_match(lw, _GARMENT_SORTED)
            fuzzy_c = _fuzzy_match(lw, _COLOR_SORTED) if not fuzzy_g else None
            canon = fuzzy_g or fuzzy_c or canon

        out.append(canon)
    return "".join(out)


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
    # canonicalize before matching: fixes plurals and typos in one pass
    normalized_query = _canonicalize_tokens(query)

    garment_attrs = _find_garment_color_pairs(normalized_query)
    scene = _find_scene(normalized_query)

    residual = normalized_query
    for garment, color in garment_attrs:
        residual = re.sub(rf"\b{re.escape(garment)}\b", "", residual, flags=re.I)
        if color:
            residual = re.sub(rf"\b{re.escape(color)}\b", "", residual, flags=re.I)

    if scene:
        for kw in config.SCENE_KEYWORDS[scene]:
            residual = re.sub(rf"\b{re.escape(kw)}\b", "", residual, flags=re.I)

    residual = re.sub(r"\s+", " ", residual).strip(" .,")

    meaningful_tokens = [
        tok for tok in re.findall(r"[a-zA-Z']+", residual.lower())
        if tok not in _STOPWORDS
    ]
    vibe = " ".join(meaningful_tokens) if meaningful_tokens else None

    return ParsedQuery(garment_attrs=garment_attrs, scene=scene, vibe=vibe, raw_query=query)