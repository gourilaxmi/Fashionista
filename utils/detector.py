from dataclasses import dataclass
from typing import List

import data.config as config


@dataclass
class Detection:
    label: str
    box: tuple  
    score: float

# chunking as grounding dino produces less when chunk size is high
def _chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


class GroundingDinoDetector:
    def __init__(self, model_name: str = config.GROUNDING_DINO_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    def _lazy_load(self):
        if self._model is not None:
            return
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_name)
        self._model.to(self.device).eval()

    def detect(self, image, vocab: List[str] = None, chunk_size: int = 16) -> List[Detection]:
        
        import torch

        self._lazy_load()
        vocab = vocab or config.GARMENT_VOCAB
        all_detections: List[Detection] = []

        for chunk in _chunked(vocab, chunk_size):
            text_prompt = ". ".join(chunk) + "."

            inputs = self._processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self._model(**inputs)

            results = self._processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                box_threshold=config.DETECTION_BOX_THRESHOLD,
                text_threshold=config.DETECTION_TEXT_THRESHOLD,
                target_sizes=[image.size[::-1]],
            )[0]

            for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
                label_clean = label.strip()
                if label_clean not in chunk:
                    continue  
                all_detections.append(Detection(
                    label=label_clean,
                    box=tuple(box.tolist()),
                    score=float(score),
                ))
        return all_detections

    def ground_phrase(self, image, phrase: str) -> float:
        # how confidently does the phrase ground onto this specific image
        dets = self.detect(image, vocab=[phrase], chunk_size=1)
        return max((d.score for d in dets), default=0.0)


def get_detector():
    return GroundingDinoDetector()