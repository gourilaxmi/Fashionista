import numpy as np

import data.config as config


class SamSegmenter:
    def __init__(self, checkpoint_type: str = config.SAM_CHECKPOINT_TYPE, device: str = "cpu"):
        self.checkpoint_type = checkpoint_type
        self.device = device
        self._predictor = None

    # load SAM model
    def _lazy_load(self):
        if self._predictor is not None:
            return
        if getattr(self, "_load_failed", False):
            raise RuntimeError("SAM checkpoint failed to load previously; skipping.")
        from segment_anything import sam_model_registry, SamPredictor
        checkpoint_paths = {
    "vit_b": "checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": "checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": "checkpoints/sam_vit_h_4b8939.pth",
}
        try:
            sam = sam_model_registry[self.checkpoint_type](checkpoint=checkpoint_paths[self.checkpoint_type])
        except Exception:
            self._load_failed = True
            raise
        sam.to(self.device)
        self._predictor = SamPredictor(sam)

    def segment(self, image, box) -> np.ndarray:
        # return a boolean mask of the same size as the image, where True = inside the box
        self._lazy_load()
        self._predictor.set_image(np.array(image))
        box_arr = np.array(box)
        masks, scores, _ = self._predictor.predict(box=box_arr, multimask_output=False)
        return masks[0]


def get_segmenter():
    return SamSegmenter()
