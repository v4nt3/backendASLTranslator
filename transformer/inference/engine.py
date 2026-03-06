import torch # type: ignore
import torch.nn.functional as F # type: ignore
import numpy as np # type: ignore
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from transformer.core.config import Config, get_pose_only_config
from transformer.core.exceptions import InferenceError, ModelError
from transformer.model.transformer import create_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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


@dataclass
class SentencePrediction:
    """Sentence prediction result (multiple signs)."""
    signs: List[SignPrediction] = field(default_factory=list)
    sentence: str = ""
    total_frames: int = 0
    processing_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "signs": [s.to_dict() for s in self.signs],
            "sentence": self.sentence,
            "total_frames": self.total_frames,
            "processing_time": round(self.processing_time, 4),
        }



class SignLanguageInference:
    """
    Stateless inference engine.

    Receives pre-extracted keypoints (from the frontend via MediaPipe JS)
    and runs the Transformer model to produce sign predictions.

    The model is loaded ONCE when the engine is created (at app startup)
    and reused for every request.
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

        # --- Load config ---
        if config_path:
            self.config = Config.from_yaml(config_path)
        else:
            self.config = self._load_config_from_checkpoint(checkpoint_path)

        # --- Load model ---
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        # --- Load labels ---
        self.idx_to_label = self._load_labels(labels_path)

        # --- Shortcuts ---
        self.inf_config = self.config.inference
        self.max_seq_length = self.config.data.max_seq_length

        logger.info(
            f"Inference engine ready: device={self.device}, "
            f"classes={len(self.idx_to_label)}, "
            f"max_seq_length={self.max_seq_length}"
        )

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_config_from_checkpoint(self, checkpoint_path: str) -> Config:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if "config" in checkpoint:
            return Config.from_dict(checkpoint["config"])
        logger.warning("No config in checkpoint, using default pose-only config")
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
            logger.info("Model loaded (strict=True)")
        except RuntimeError as e:
            logger.warning(f"Strict load failed: {e}")
            model.load_state_dict(state_dict, strict=False)
            logger.info("Model loaded (strict=False)")

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
        Pad/truncate to max_seq_length and build attention mask.

        Args:
            features: [seq_len, feature_dim]

        Returns:
            (pose_tensor [1, max_seq, dim], mask_tensor [1, max_seq])
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
            torch.tensor(features, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )
        mask_tensor = (
            torch.tensor(mask, dtype=torch.bool)
            .unsqueeze(0)
            .to(self.device)
        )
        return pose_tensor, mask_tensor


    def _predict_window(self, features: np.ndarray) -> SignPrediction:
        """Run model on a single window of features -> SignPrediction."""
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


    def _frame_has_any_hand(self, frame_features: np.ndarray) -> bool:
        """Check if a single frame has at least one hand detected."""
        left = frame_features[:63]
        right = frame_features[63:126]
        return bool(
            np.any(np.abs(left) > self.DETECTION_EPS)
            or np.any(np.abs(right) > self.DETECTION_EPS)
        )

    def _first_valid_hand_frame(self, features: np.ndarray) -> Optional[int]:
        """Find first frame index where at least one hand is present."""
        if features is None or len(features) == 0:
            return None
        left_valid = np.any(np.abs(features[:, :63]) > self.DETECTION_EPS, axis=1)
        right_valid = np.any(
            np.abs(features[:, 63:126]) > self.DETECTION_EPS, axis=1
        )
        any_valid = left_valid | right_valid
        if not any_valid.any():
            return None
        return int(np.argmax(any_valid))

    def _window_has_valid_hands(
        self, window: np.ndarray, min_ratio: float = 0.3
    ) -> bool:
        """Check if enough frames in a window have hand keypoints."""
        if window is None or len(window) == 0:
            return False
        left_valid = np.any(np.abs(window[:, :63]) > self.DETECTION_EPS, axis=1)
        right_valid = np.any(
            np.abs(window[:, 63:126]) > self.DETECTION_EPS, axis=1
        )
        return float(np.mean(left_valid | right_valid)) >= min_ratio


    def predict_sign(self, features: np.ndarray) -> SignPrediction:
        """
        Predict a single isolated sign from a sequence of keypoint features.

        Args:
            features: numpy array [num_frames, 858]
                      (already extracted from MediaPipe on the client)

        Returns:
            SignPrediction
        """
        start = time.time()

        first_idx = self._first_valid_hand_frame(features)
        if first_idx is not None:
            features = features[first_idx:]
        else:
            first_idx = 0

        prediction = self._predict_window(features)
        prediction.start_frame = first_idx
        prediction.end_frame = first_idx + len(features)

        elapsed = time.time() - start
        logger.info(
            f"predict_sign: '{prediction.label}' "
            f"(conf={prediction.confidence:.3f}, "
            f"frames={len(features)}, time={elapsed:.3f}s)"
        )
        return prediction


    def predict_sentence(self, features: np.ndarray) -> SentencePrediction:
        """
        Predict a sequence of signs (sentence) via sliding window.

        Args:
            features: numpy array [num_frames, 858]

        Returns:
            SentencePrediction
        """
        start_time = time.time()
        original_total = len(features)

        first_idx = self._first_valid_hand_frame(features)
        if first_idx is not None:
            features = features[first_idx:]
        else:
            first_idx = 0

        total = len(features)
        window_size = self.inf_config.window_size
        stride = self.inf_config.window_stride
        min_conf = self.inf_config.min_confidence

        raw: List[SignPrediction] = []

        for s in range(0, total, stride):
            e = min(s + window_size, total)
            window = features[s:e]

            if len(window) < self.config.data.min_seq_length:
                continue
            if not self._window_has_valid_hands(window):
                continue

            pred = self._predict_window(window)
            pred.start_frame = s + first_idx
            pred.end_frame = e + first_idx

            if pred.confidence >= min_conf:
                raw.append(pred)

        merged = self._merge_predictions(raw) if self.inf_config.merge_duplicates else raw
        sentence = " ".join(p.label for p in merged)
        elapsed = time.time() - start_time

        result = SentencePrediction(
            signs=merged,
            sentence=sentence,
            total_frames=original_total,
            processing_time=elapsed,
        )

        logger.info(
            f"predict_sentence: '{sentence}' "
            f"({len(merged)} signs, {original_total} frames, {elapsed:.2f}s)"
        )
        return result


    def _merge_predictions(
        self, predictions: List[SignPrediction]
    ) -> List[SignPrediction]:
        if not predictions:
            return []

        merged = [predictions[0]]
        for pred in predictions[1:]:
            if pred.label == merged[-1].label:
                if pred.confidence > merged[-1].confidence:
                    merged[-1] = SignPrediction(
                        label=pred.label,
                        confidence=pred.confidence,
                        top_k=pred.top_k,
                        start_frame=merged[-1].start_frame,
                        end_frame=pred.end_frame,
                    )
                else:
                    merged[-1].end_frame = pred.end_frame
            else:
                merged.append(pred)
        return merged
