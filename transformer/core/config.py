"""
Configuration system for sign language recognition.

Supports loading from YAML files and checkpoint dictionaries.
Optimized for CPU inference on Render free tier (~512MB RAM).
"""

import yaml # type: ignore
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple


class FeatureType(Enum):
    VISUAL = "visual"
    POSE = "pose"
    MULTIMODAL = "multimodal"


class OptimizerType(Enum):
    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"


class SchedulerType(Enum):
    COSINE_WARMUP = "cosine_warmup"
    COSINE = "cosine"
    PLATEAU = "plateau"
    ONE_CYCLE = "one_cycle"


@dataclass
class DataConfig:
    data_dir: str = "data/dataset"
    features_dir: str = "data/features"
    labels_file: str = "data/labels.json"
    num_classes: int = 2286
    max_seq_length: int = 64
    min_seq_length: int = 8
    visual_feature_dim: int = 2048
    pose_feature_dim: int = 858
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    normalize_pose: bool = True
    add_velocity: bool = False
    add_acceleration: bool = False


@dataclass
class AugmentationConfig:
    enabled: bool = True
    temporal_crop_prob: float = 0.3
    temporal_crop_ratio: Tuple[float, float] = (0.85, 1.0)
    speed_augment_prob: float = 0.3
    speed_range: Tuple[float, float] = (0.85, 1.15)
    temporal_mask_prob: float = 0.15
    temporal_mask_ratio: float = 0.05
    pose_noise_prob: float = 0.2
    pose_noise_std: float = 0.005
    pose_dropout_prob: float = 0.1
    pose_dropout_ratio: float = 0.03
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 0.0
    mix_prob: float = 0.3


@dataclass
class ModelConfig:
    model_type: str = "transformer"
    feature_type: FeatureType = FeatureType.POSE
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ff_dim: int = 1536
    dropout: float = 0.2
    attention_dropout: float = 0.1
    path_dropout: float = 0.1
    visual_proj_dim: int = 512
    pose_proj_dim: int = 512
    use_cross_modal_attention: bool = False
    cross_modal_layers: int = 2
    classifier_hidden_dim: int = 1024
    use_pooling: str = "attention"
    use_learnable_pos_encoding: bool = True
    max_position_embeddings: int = 128


@dataclass
class TrainingConfig:
    batch_size: int = 128
    eval_batch_size: int = 256
    num_epochs: int = 150
    optimizer: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 5e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.05
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    scheduler: SchedulerType = SchedulerType.COSINE_WARMUP
    warmup_epochs: int = 8
    warmup_ratio: float = 0.06
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001
    save_top_k: int = 3
    checkpoint_dir: str = "outputs/checkpoints"
    save_every_n_epochs: int = 10
    use_amp: bool = True
    amp_dtype: str = "float16"
    use_class_weights: bool = True
    class_weight_power: float = 0.5
    use_balanced_sampling: bool = False
    label_smoothing: float = 0.1
    seed: int = 42
    deterministic: bool = False

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


@dataclass
class InferenceConfig:
    window_size: int = 32
    window_stride: int = 16
    min_confidence: float = 0.15
    merge_duplicates: bool = True
    min_gap_frames: int = 8
    top_k: int = 5
    camera_fps: int = 30
    camera_buffer_seconds: float = 3.0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    experiment_name: str = "sign_language_pose_only"
    output_dir: str = "outputs"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls._from_raw_dict(raw)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """Load configuration from a plain dictionary (e.g. from a checkpoint)."""
        return cls._from_raw_dict(d)

    @classmethod
    def _from_raw_dict(cls, raw: Dict[str, Any]) -> "Config":
        """Build Config from a raw dictionary, handling enum conversions."""

        data_cfg = DataConfig(**raw.get("data", {}))

        # Augmentation may store tuples as lists
        aug_raw = raw.get("augmentation", {})
        for key in ("temporal_crop_ratio", "speed_range"):
            if key in aug_raw and isinstance(aug_raw[key], list):
                aug_raw[key] = tuple(aug_raw[key])
        aug_cfg = AugmentationConfig(**aug_raw)

        # Model - convert feature_type string to enum
        model_raw = raw.get("model", {})
        if "feature_type" in model_raw and isinstance(model_raw["feature_type"], str):
            model_raw["feature_type"] = FeatureType(model_raw["feature_type"])
        model_cfg = ModelConfig(**model_raw)

        # Training - convert enums
        train_raw = raw.get("training", {})
        if "optimizer" in train_raw and isinstance(train_raw["optimizer"], str):
            train_raw["optimizer"] = OptimizerType(train_raw["optimizer"])
        if "scheduler" in train_raw and isinstance(train_raw["scheduler"], str):
            train_raw["scheduler"] = SchedulerType(train_raw["scheduler"])
        train_cfg = TrainingConfig(**train_raw)

        inf_cfg = InferenceConfig(**raw.get("inference", {}))

        return cls(
            data=data_cfg,
            augmentation=aug_cfg,
            model=model_cfg,
            training=train_cfg,
            inference=inf_cfg,
            experiment_name=raw.get("experiment_name", "sign_language_pose_only"),
            output_dir=raw.get("output_dir", "outputs"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire config to a JSON-safe dictionary."""
        d = asdict(self)
        # Enums -> values
        d["model"]["feature_type"] = self.model.feature_type.value
        d["training"]["optimizer"] = self.training.optimizer.value
        d["training"]["scheduler"] = self.training.scheduler.value
        return d


def get_pose_only_config() -> Config:
    """Return a sensible default config for pose-only mode."""
    return Config(
        data=DataConfig(
            num_classes=2286,
            max_seq_length=64,
            min_seq_length=8,
            pose_feature_dim=858,
        ),
        model=ModelConfig(
            feature_type=FeatureType.POSE,
            hidden_dim=512,
            num_layers=6,
            num_heads=8,
            ff_dim=1536,
        ),
        inference=InferenceConfig(
            window_size=32,
            window_stride=16,
            min_confidence=0.15,
            top_k=5,
        ),
    )
