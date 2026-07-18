import numpy as np

import data.config as config


def _as_tensor(feats):
    if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
        return feats.pooler_output
    if hasattr(feats, "image_embeds") and feats.image_embeds is not None:
        return feats.image_embeds
    if hasattr(feats, "text_embeds") and feats.text_embeds is not None:
        return feats.text_embeds
    if hasattr(feats, "last_hidden_state"):
        # fall back to the [CLS]/pooled token of the last hidden state
        return feats.last_hidden_state[:, 0, :]
    return feats  


class FashionClipEmbedder:
    def __init__(self, model_name: str = config.FASHIONCLIP_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    def _lazy_load(self):
        if self._model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
        self._processor = CLIPProcessor.from_pretrained(self.model_name)

    def embed_image(self, image) -> np.ndarray:
    #Same embedding space as embed_image -- used for both free-text vibe
    # queries and scene-classification prompts
        import torch
        self._lazy_load()
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        feats = _as_tensor(feats)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()[0]

    def embed_text(self, text: str) -> np.ndarray:
        import torch
        self._lazy_load()
        inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        feats = _as_tensor(feats)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()[0]


def get_embedder():
    return FashionClipEmbedder()

embedder = get_embedder