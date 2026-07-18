from PIL import Image
import numpy as np

import data.config as config


def nearest_color(rgb: np.ndarray) -> str:
    # find the nearest color name in the palette from web colors
    best_name, best_dist = None, float("inf")
    for name, ref_rgb in config.COLOR_PALETTE.items():
        dist = np.linalg.norm(rgb - np.array(ref_rgb))
        if dist < best_dist:
            best_dist, best_name = dist, name
    return best_name


def extract_color(image, mask: np.ndarray, k: int = 3) -> str:
    # extract the dominant color from the masked region of the image
    arr = np.array(image.convert("RGB"))
    pixels = arr[mask]
    if pixels.size == 0:
        return "unknown"

    if len(pixels) > 2000:  # subsample for speed on large masks
        idx = np.random.choice(len(pixels), 2000, replace=False)
        pixels = pixels[idx]
    # cluster the pixels to find the dominant color, using KMeans 
    try:
        from sklearn.cluster import KMeans
        k_eff = min(k, len(np.unique(pixels, axis=0)))
        if k_eff < 1:
            return "unknown"
        km = KMeans(n_clusters=k_eff, n_init=3, random_state=0).fit(pixels)
        counts = np.bincount(km.labels_)
        dominant_rgb = km.cluster_centers_[np.argmax(counts)]
    except ImportError:
        dominant_rgb = pixels.mean(axis=0)

    return nearest_color(dominant_rgb)
