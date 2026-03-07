import torch #type: ignore
import torch.nn.functional as F #type: ignore
import numpy as np #type: ignore
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from transformer.core.config import Config, get_pose_only_config
from transformer.core.exceptions import InferenceError, ModelError
from transformer.model.transformer import create_model

logger = logging.getLogger(__name__)


@dataclass
class SignPrediction:
    """Single sign prediction result."""
    label: str
    confidence: float
    top_k: List[Tuple[str, float]] = field(default_factory=list)
    start_frame: int = 0
    end_frame: int = 0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "top_k": [
                {"label": l, "confidence": round(c, 4)} for l, c in self.top_k
            ],
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }


class SignLanguageInference:
    """
    Stateless inference engine.

    Receives pre-extracted keypoints (from the frontend via MediaPipe JS)
    and runs the Transformer model to produce sign predictions.

    The model is loaded ONCE at app startup and reused for every request.
    """

    DETECTION_EPS = 1e-4

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        labels_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.config = (
            Config.from_yaml(config_path)
            if config_path
            else self._load_config_from_checkpoint(checkpoint_path)
        )

        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        self.idx_to_label = self._load_labels(labels_path)

        self.inf_config = self.config.inference
        self.max_seq_length = self.config.data.max_seq_length

        logger.info(
            f"Inference engine ready: device={self.device}, "
            f"classes={len(self.idx_to_label)}, "
            f"max_seq_length={self.max_seq_length}"
        )

    def _load_config_from_checkpoint(self, checkpoint_path: str) -> Config:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if "config" in checkpoint:
            return Config.from_dict(checkpoint["config"])
        logger.warning("No config found in checkpoint — using default pose-only config")
        return get_pose_only_config()

    def _load_model(self, checkpoint_path: str):
        model = create_model(self.config.model, self.config.data, self.device)

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        try:
            model.load_state_dict(state_dict, strict=True)
            logger.info("Model weights loaded (strict=True)")
        except RuntimeError:
            logger.warning(
                "Strict weight loading failed — attempting partial load (strict=False)"
            )
            model.load_state_dict(state_dict, strict=False)
            logger.info("Model weights loaded (strict=False)")

        return model.to(self.device)

    def _load_labels(self, labels_path: Optional[str]) -> Dict[int, str]:
        if labels_path is None:
            raise ModelError(
                "labels_path is required",
                recovery_hint="Provide the path to labels.json",
            )
        with open(labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            if "idx_to_label" in data:
                return {int(k): v for k, v in data["idx_to_label"].items()}
            if "label_to_idx" in data:
                return {int(v): k for k, v in data["label_to_idx"].items()}

        raise ModelError(
            "Unrecognized labels.json format. "
            "Expected 'idx_to_label' or 'label_to_idx' key."
        )

    def _prepare_features(
        self, features: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad / truncate to max_seq_length and build the attention mask.

        Args:
            features: [seq_len, feature_dim]

        Returns:
            pose_tensor  [1, max_seq_length, feature_dim]
            mask_tensor  [1, max_seq_length]  (True = valid frame)
        """
        num_frames = len(features)

        if num_frames > self.max_seq_length:
            features = features[: self.max_seq_length]
            num_frames = self.max_seq_length

        if num_frames < self.max_seq_length:
            pad_len = self.max_seq_length - num_frames
            features = np.concatenate(
                [features, np.zeros((pad_len, features.shape[-1]), dtype=np.float32)]
            )
            mask = np.concatenate(
                [np.ones(num_frames, dtype=bool), np.zeros(pad_len, dtype=bool)]
            )
        else:
            mask = np.ones(self.max_seq_length, dtype=bool)

        pose_tensor = (
            torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        mask_tensor = (
            torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(self.device)
        )
        return pose_tensor, mask_tensor

    def _first_valid_hand_frame(self, features: np.ndarray) -> Optional[int]:
        """Return the index of the first frame with at least one hand detected."""
        if features is None or len(features) == 0:
            return None
        left_valid = np.any(np.abs(features[:, :63]) > self.DETECTION_EPS, axis=1)
        right_valid = np.any(
            np.abs(features[:, 63:126]) > self.DETECTION_EPS, axis=1
        )
        any_valid = left_valid | right_valid
        return int(np.argmax(any_valid)) if any_valid.any() else None

    def _predict_window(self, features: np.ndarray) -> SignPrediction:
        """Run the model on a single window of features."""
        pose_tensor, mask_tensor = self._prepare_features(features)

        with torch.no_grad():
            output = self.model(
                pose_features=pose_tensor, attention_mask=mask_tensor
            )

        probs = F.softmax(output.logits, dim=-1)[0]
        top_k_probs, top_k_indices = torch.topk(
            probs, min(self.inf_config.top_k, len(probs))
        )

        top_k = [
            (self.idx_to_label.get(idx.item(), str(idx.item())), prob.item())
            for idx, prob in zip(top_k_indices, top_k_probs)
        ]

        return SignPrediction(
            label=top_k[0][0],
            confidence=top_k[0][1],
            top_k=top_k,
        )

    def predict_sign(self, features: np.ndarray) -> SignPrediction:
        """
        Predict a single isolated sign from a pre-extracted keypoint sequence.

        Args:
            features: [num_frames, 858]  — already extracted by the client

        Returns:
            SignPrediction
        """
        start = time.perf_counter()

        prediction = self._predict_window(features)
        prediction.start_frame = 0
        prediction.end_frame = len(features)

        elapsed = time.perf_counter() - start
        hand_offset = self._first_valid_hand_frame(features)
        logger.info(
            f"predict_sign completed: frames={len(features)}, "
            f"hand_offset={hand_offset}, latency={elapsed:.3f}s"
        )
        return prediction