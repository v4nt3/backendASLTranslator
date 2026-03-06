from pydantic import BaseModel, Field
from typing import List, Optional


class PredictSignRequest(BaseModel):
    """Request body for single sign prediction."""
    keypoints: List[List[float]] = Field(
        ...,
        description=(
            "2D array of keypoints [num_frames, 858]. "
            "Each frame contains: left_hand(63) + right_hand(63) + "
            "face_subset(204) + body(99) + velocities(429) = 858 values."
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "keypoints": [
                    [0.0] * 858,
                    [0.0] * 858,
                ]
            }
        }


class PredictSentenceRequest(BaseModel):
    """Request body for sentence prediction (sliding window)."""
    keypoints: List[List[float]] = Field(
        ...,
        description="2D array of keypoints [num_frames, 858].",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "keypoints": [[0.0] * 858] * 32
            }
        }


class TopKPrediction(BaseModel):
    label: str
    confidence: float


class SignPredictionResponse(BaseModel):
    label: str
    confidence: float
    top_k: List[TopKPrediction]
    start_frame: int
    end_frame: int


class PredictSignResponse(BaseModel):
    success: bool = True
    prediction: SignPredictionResponse


class SentencePredictionResponse(BaseModel):
    success: bool = True
    sentence: str
    signs: List[SignPredictionResponse]
    total_frames: int
    processing_time: float


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    device: str
    num_classes: int
    max_seq_length: int


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None
