# ASL Translator Backend API

A high-performance REST API for American Sign Language (ASL) recognition using deep learning. This backend processes keypoint sequences extracted through MediaPipe and returns sign language predictions using a Transformer-based neural network model.

## Features

- **Transformer-based Sign Recognition**: State-of-the-art deep learning model with 2,286 ASL sign classes
- **Secure API**: API key authentication and rate limiting for production deployments
- **Real-time Inference**: Optimized inference engine with concurrent request handling
- **LLM Integration**: Sentence processing and contextualization using Google Generative AI

## Tech Stack

- **Framework**: FastAPI
- **Deep Learning**: PyTorch with Transformer architecture
- **Language Model**: Google Generative AI
- **Server**: Uvicorn (ASGI)
- **Keypoint Detection**: MediaPipe (client-side extraction)

## Project Structure

```
backendASLTranslator/
├── app/
│   ├── main.py              # FastAPI application and endpoints
│   ├── llm_service.py       # LLM integration for sentence processing
│   ├── schemas.py           # Pydantic request/response models
│   └── security.py          # Authentication and rate limiting
├── transformer/
│   ├── core/
│   │   ├── config.py        # Configuration management
│   │   └── exceptions.py    # Custom exceptions
│   ├── inference/
│   │   └── engine.py        # Sign language inference engine
│   └── model/
│       ├── components.py    # Model components
│       └── transformer.py   # Transformer architecture
├── models/
│   ├── best_model.pt        # Trained model checkpoint (2,286 classes)
│   └── labels.json          # Sign class labels
├── tests/
│   └── test_units.py        # Unit tests
├── config.yaml              # Model configuration
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker image configuration
├── pytest.ini               # Pytest configuration
└── Procfile                 # Heroku deployment configuration
```

## Installation

### Prerequisites

- Python 3.10+
- PyTorch (CPU or CUDA)
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd backendASLTranslator
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download model (if not included)**
   ```bash
   python download_model.py
   ```

5. **Set environment variables**
   ```bash
   # Create .env file
   API_KEY=your_secret_api_key
   ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173,http://localhost:8080
   DEVICE=cpu  # or 'cuda' for GPU
   GOOGLE_API_KEY=your_google_api_key  # For LLM features
   ```

## Usage

### Running the Server

```bash
# Development
python -m uvicorn app.main:app --reload

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Interactive API documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## API Endpoints

### System Endpoints

#### Health Check
```http
GET /health
```

Returns server status and model information. **No authentication required**.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu",
  "num_classes": 2286,
  "max_seq_length": 64
}
```

#### Get Labels
```http
GET /labels
```

Returns all available sign language labels.

**Response:**
```json
{
  "num_classes": 2286,
  "labels": {
    "0": "hello",
    "1": "goodbye",
    ...
  }
}
```

#### Get Configuration
```http
GET /config
```

Returns active inference configuration for debugging.

### Prediction Endpoints

#### Predict Single Sign
```http
POST /predict/sign
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
  "keypoints": [
    [x1, y1, z1, conf1, x2, y2, z2, conf2, ...],
    [x1, y1, z1, conf1, x2, y2, z2, conf2, ...],
    ...
  ]
}
```

**Parameters:**
- `keypoints`: 2D array of shape (num_frames, 858)
  - Left hand: 63 features
  - Right hand: 63 features
  - Face: 204 features
  - Body: 99 features
  - Velocity: 429 features
- Maximum frames per request: 512

**Response:**
```json
{
  "prediction": {
    "label": "hello",
    "confidence": 0.9876,
    "top_k": [
      {"label": "hello", "confidence": 0.9876},
      {"label": "hi", "confidence": 0.0098},
      {"label": "greetings", "confidence": 0.0015}
    ],
    "start_frame": 5,
    "end_frame": 42
  }
}
```

#### Process Sentence
```http
POST /process/sentence
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
  "words": ["hello", "how", "are", "you"]
}
```

Processes a list of words through the LLM for contextualization and sentence creation.

**Response:**
```json
{
  "sentence": "Hello, how are you?"
}
```

## Model Configuration

The model is configured through `config.yaml`. Key parameters:

```yaml
data:
  num_classes: 2286           # Total ASL signs
  max_seq_length: 64          # Maximum sequence length
  pose_feature_dim: 858       # Input feature dimension

model:
  model_type: transformer
  hidden_dim: 512
  num_layers: 4
  num_heads: 8
  ff_dim: 2048
  dropout: 0.2
```

## Authentication & Rate Limiting

### API Key Authentication

All protected endpoints require an `Authorization` header:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:8000/predict/sign
```

### Rate Limiting

- Default: 30 requests per minute per API key
- Configurable via environment variables

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | Required | Secret key for API authentication |
| `GOOGLE_API_KEY` | Required | Google API key for LLM features |
| `ALLOWED_ORIGINS` | Localhost | CORS allowed origins (comma-separated) |
| `DEVICE` | `cpu` | Compute device (`cpu` or `cuda`) |
| `CHECKPOINT_PATH` | `models/best_model.pt` | Path to model checkpoint |
| `CONFIG_PATH` | `config.yaml` | Path to configuration file |
| `LABELS_PATH` | `models/labels.json` | Path to labels file |

## Performance Considerations

- **Inference Time**: ~50-100ms per sign prediction (depends on sequence length)
- **Concurrent Requests**: Supports 2 concurrent inference requests by default
- **Memory Usage**: ~2GB typical (trained on GPU, runs on CPU)
- **Maximum Frames**: 512 frames per request

## Input Data Format

### Keypoint Extraction

Keypoints should be extracted using MediaPipe's Holistic solution with the following structure:

1. **Left Hand**: 21 landmarks × 3 coordinates = 63 features
2. **Right Hand**: 21 landmarks × 3 coordinates = 63 features
3. **Face**: 468 landmarks → pooled to 204 features
4. **Body**: 33 landmarks × 3 coordinates = 99 features
5. **Velocity**: Temporal derivatives of all keypoints = 429 features

Total: 858 features per frame

## Error Handling

The API returns standardized error responses:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common HTTP Status Codes:**
- `200 OK`: Successful request
- `400 Bad Request`: Invalid input data
- `401 Unauthorized`: Missing or invalid API key
- `413 Payload Too Large`: Input exceeds maximum frame limit
- `422 Unprocessable Entity`: Invalid data format
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error during inference
- `503 Service Unavailable`: Model not loaded yet