from pydantic import BaseModel, Field #type: ignore
from typing import List, Optional


class PredictSignRequest(BaseModel):
    """Request body for single sign prediction."""
    keypoints: List[List[float]] = Field(
        ...,
        description=(
            "2D array of keypoints [num_frames, 858]. "
            "Each frame: left_hand(63) + right_hand(63) + "
            "face_subset(204) + body(99) + velocities(429) = 858 values. "
            "Extracted client-side with MediaPipe Holistic."
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "keypoints": [[0.0] * 858, [0.0] * 858]
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

class ProcessSentenceRequest(BaseModel):
    words: str

class ProcessSentenceResponse(BaseModel):
    sentence: str