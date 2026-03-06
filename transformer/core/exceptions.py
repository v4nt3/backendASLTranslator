from typing import Optional, Dict, Any


class SignLanguageError(Exception):
    """Base exception for all sign language system errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        recovery_hint: Optional[str] = None
    ):
        super().__init__(message)
        self.details = details or {}
        self.recovery_hint = recovery_hint


class ConfigError(SignLanguageError):
    """Configuration-related errors."""
    pass


class DataLoadError(SignLanguageError):
    """Data loading errors."""
    pass


class FeatureExtractionError(SignLanguageError):
    """Feature extraction errors."""
    pass


class ModelError(SignLanguageError):
    """Model loading or inference errors."""
    pass


class TrainingError(SignLanguageError):
    """Training-related errors."""
    pass


class InferenceError(SignLanguageError):
    """Inference pipeline errors."""
    pass
