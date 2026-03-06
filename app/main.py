import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas import (
    PredictSignRequest,
    PredictSignResponse,
    PredictSentenceRequest,
    SentencePredictionResponse,
    HealthResponse,
    ErrorResponse,
    SignPredictionResponse,
    TopKPrediction,
)
from app.security import (
    verify_api_key,
    check_rate_limit,
    acquire_inference_slot,
    release_inference_slot,
    MAX_FRAMES_PER_REQUEST,
    EXPECTED_FEATURE_DIM,
)
from transformer.inference.engine import SignLanguageInference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

engine: Optional[SignLanguageInference] = None

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "models/best_model.pt")
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")
LABELS_PATH = os.getenv("LABELS_PATH", "models/labels.json")
DEVICE = os.getenv("DEVICE", "cpu")

# Allowed origins for CORS
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:8080",
).split(",")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine

    logger.info("Loading model...")
    start = time.time()

    try:
        engine = SignLanguageInference(
            checkpoint_path=CHECKPOINT_PATH,
            config_path=CONFIG_PATH if os.path.exists(CONFIG_PATH) else None,
            labels_path=LABELS_PATH,
            device=DEVICE,
        )
        elapsed = time.time() - start
        logger.info(f"Model loaded in {elapsed:.2f}s")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

    yield  # App is running

    logger.info("Shutting down...")
    engine = None

app = FastAPI(
    title="Sign Language Recognition API",
    description=(
        "API REST para reconocimiento de lengua de senas. "
        "Recibe secuencias de keypoints (pre-extraidos con MediaPipe en el "
        "navegador) y retorna predicciones del modelo Transformer."
    ),
    version="1.0.0",
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse},
    },
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _validate_keypoints(keypoints: list) -> np.ndarray:
    """
    Convert and validate keypoints from request body.

    Security checks (STRIDE - Tampering / DoS):
    - 2D shape validation
    - Feature dimension must be exactly EXPECTED_FEATURE_DIM (858)
    - Max frames limited to MAX_FRAMES_PER_REQUEST (default 512)
    - Non-empty array
    """
    arr = np.array(keypoints, dtype=np.float32)

    if arr.ndim != 2:
        raise HTTPException(
            status_code=422,
            detail=f"keypoints must be a 2D array, got shape {arr.shape}",
        )

    if arr.shape[0] == 0:
        raise HTTPException(status_code=422, detail="keypoints array is empty")

    if arr.shape[0] > MAX_FRAMES_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Input too large. Maximum {MAX_FRAMES_PER_REQUEST} frames "
                f"allowed per request, got {arr.shape[0]}."
            ),
        )

    if arr.shape[1] != EXPECTED_FEATURE_DIM:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Each frame must have {EXPECTED_FEATURE_DIM} values, "
                f"got {arr.shape[1]}. Expected: left_hand(63) + "
                f"right_hand(63) + face(204) + body(99) + velocity(429) = 858"
            ),
        )

    return arr


def _ensure_engine():
    """Raise 503 if the model is not loaded."""
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server is starting up.",
        )

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and model status."""
    if engine is None:
        return HealthResponse(
            status="loading",
            model_loaded=False,
            device="unknown",
            num_classes=0,
            max_seq_length=0,
        )
    return HealthResponse(
        status="ok",
        model_loaded=True,
        device=str(engine.device),
        num_classes=len(engine.idx_to_label),
        max_seq_length=engine.max_seq_length,
    )


@app.post(
    "/predict/sign",
    response_model=PredictSignResponse,
    tags=["Prediction"],
    summary="Predict a single isolated sign",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def predict_sign(request: PredictSignRequest):
    """
    Predict a single sign from a sequence of keypoint frames.

    The frontend extracts keypoints with MediaPipe Holistic in the browser
    and sends them as a 2D array [num_frames, 858].

    Security: requires X-API-Key header, rate-limited, concurrency-controlled.
    """
    _ensure_engine()
    features = _validate_keypoints(request.keypoints)

    await acquire_inference_slot()
    try:
        prediction = engine.predict_sign(features)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        release_inference_slot()

    return PredictSignResponse(
        prediction=SignPredictionResponse(
            label=prediction.label,
            confidence=round(prediction.confidence, 4),
            top_k=[
                TopKPrediction(label=l, confidence=round(c, 4))
                for l, c in prediction.top_k
            ],
            start_frame=prediction.start_frame,
            end_frame=prediction.end_frame,
        )
    )

@app.post(
    "/predict/sentence",
    response_model=SentencePredictionResponse,
    tags=["Prediction"],
    summary="Predict a sentence (sequence of signs) via sliding window",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def predict_sentence(request: PredictSentenceRequest):
    """
    Predict a sequence of signs from keypoint frames using sliding window.

    Pipeline: sliding window -> inference -> softmax -> top-k ->
    confidence filter -> merge duplicates -> sentence.

    Security: requires X-API-Key header, rate-limited, concurrency-controlled.
    """
    _ensure_engine()
    features = _validate_keypoints(request.keypoints)

    await acquire_inference_slot()
    try:
        result = engine.predict_sentence(features)
    except Exception as e:
        logger.error(f"Sentence prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        release_inference_slot()

    return SentencePredictionResponse(
        sentence=result.sentence,
        signs=[
            SignPredictionResponse(
                label=s.label,
                confidence=round(s.confidence, 4),
                top_k=[
                    TopKPrediction(label=l, confidence=round(c, 4))
                    for l, c in s.top_k
                ],
                start_frame=s.start_frame,
                end_frame=s.end_frame,
            )
            for s in result.signs
        ],
        total_frames=result.total_frames,
        processing_time=round(result.processing_time, 4),
    )


@app.get("/labels", tags=["System"], summary="Get all available sign labels")
async def get_labels():
    """Return the full mapping of class indices to sign labels."""
    _ensure_engine()
    return {
        "num_classes": len(engine.idx_to_label),
        "labels": engine.idx_to_label,
    }


@app.get("/config", tags=["System"], summary="Get model/inference configuration")
async def get_config():
    """Return the active configuration (useful for debugging)."""
    _ensure_engine()
    return {
        "inference": {
            "window_size": engine.inf_config.window_size,
            "window_stride": engine.inf_config.window_stride,
            "min_confidence": engine.inf_config.min_confidence,
            "merge_duplicates": engine.inf_config.merge_duplicates,
            "top_k": engine.inf_config.top_k,
            "max_seq_length": engine.max_seq_length,
        },
        "model": {
            "feature_type": engine.config.model.feature_type.value,
            "hidden_dim": engine.config.model.hidden_dim,
            "num_layers": engine.config.model.num_layers,
            "num_heads": engine.config.model.num_heads,
            "ff_dim": engine.config.model.ff_dim,
        },
        "device": str(engine.device),
    }
