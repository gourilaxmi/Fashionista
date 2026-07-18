import numpy as np

import data.config as data_config


class SceneClassifier:
    def __init__(self, embedder):
        self.embedder = embedder
        self._scene_embeddings = None

    def _ensure_scene_embeddings(self):
    # Best-scoring scene, argmax over cosine similarity to each scene's mean prompt embedding. 
    # Includes runway/studio as distractor classes
        if self._scene_embeddings is None:
            self._scene_embeddings = {}
            for scene, prompts in data_config.SCENE_PROMPT_TEMPLATES.items():
                embs = np.stack([self.embedder.embed_text(p) for p in prompts])
                mean_emb = embs.mean(axis=0)
                mean_emb = mean_emb / np.linalg.norm(mean_emb)
                self._scene_embeddings[scene] = mean_emb

    def classify(self, image):
        self._ensure_scene_embeddings()
        img_emb = self.embedder.embed_image(image)
        best_scene, best_score = None, -1.0
        for scene, scene_emb in self._scene_embeddings.items():
            score = float(np.dot(img_emb, scene_emb))
            if score > best_score:
                best_scene, best_score = scene, score
        return best_scene, best_score


def scene_classifier(embedder=None):
    return SceneClassifier(embedder)