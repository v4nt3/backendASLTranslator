# API Backend de Traductor ASL

API REST para el reconocimiento del Lenguaje de Signos Americano (ASL) utilizando aprendizaje profundo. Este backend procesa secuencias de puntos clave extraídos a través de MediaPipe y devuelve predicciones de lenguaje de signos utilizando un modelo de red neuronal basada en Transformer.

## Características

- **Reconocimiento de Signos basado en Transformer**: Modelo de aprendizaje profundo con 2,286 clases de signos ASL
- **Seguridad**: Autenticación por clave API y limitación de velocidad para implementaciones en producción
- **Inferencia en Tiempo Real**: Motor de inferencia optimizado con manejo de solicitudes concurrentes
- **Integración de LLM**: Procesamiento de oraciones y contextualización usando Google Generative AI

## Pila Tecnológica

- **Framework**: FastAPI
- **Aprendizaje Profundo**: PyTorch con arquitectura Transformer
- **Modelo de Lenguaje**: Google Generative AI
- **Servidor**: Uvicorn (ASGI)
- **Detección de Puntos Clave**: MediaPipe (extracción del lado del cliente)

## Estructura del Proyecto

```
backendASLTranslator/
├── app/
│   ├── main.py              # Aplicación FastAPI y puntos finales
│   ├── llm_service.py       # Integración LLM para procesamiento de oraciones
│   ├── schemas.py           # Modelos de solicitud/respuesta de Pydantic
│   └── security.py          # Autenticación y limitación de velocidad
├── transformer/
│   ├── core/
│   │   ├── config.py        # Gestión de configuración
│   │   └── exceptions.py    # Excepciones personalizadas
│   ├── inference/
│   │   └── engine.py        # Motor de inferencia del lenguaje de signos
│   └── model/
│       ├── components.py    # Componentes del modelo
│       └── transformer.py   # Arquitectura Transformer
├── models/
│   ├── best_model.pt        # Punto de control del modelo entrenado (2,286 clases)
│   └── labels.json          # Etiquetas de clases de signos
├── tests/
│   └── test_units.py        # Pruebas unitarias
├── config.yaml              # Configuración del modelo
├── requirements.txt         # Dependencias de Python
└── Procfile                 # Configuración de despliegue de Heroku
```

## Instalación

### Requisitos Previos

- Python 3.10+
- PyTorch (CPU o CUDA)
- pip

### Configuración

1. **Clonar el repositorio**
   ```bash
   git clone <repository-url>
   cd backendASLTranslator
   ```

2. **Crear y activar entorno virtual**
   ```bash
   python -m venv env
   env\Scripts\activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Descargar el modelo (si no está incluido)**
   ```bash
   python download_model.py
   ```

5. **Establecer variables de entorno**
   ```bash
   # Crear archivo .env
   API_KEY=your_secret_api_key
   ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173,http://localhost:8080
   DEVICE=cpu  # o 'cuda' para GPU
   GOOGLE_API_KEY=your_google_api_key  # Para características de LLM
   ```

## Uso

### Ejecutar el Servidor

```bash
# Desarrollo
python -m uvicorn app.main:app --reload

# Producción
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

La API estará disponible en `http://localhost:8000`

### Documentación de API

La documentación interactiva de la API está disponible en:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Puntos Finales de la API

### Puntos Finales del Sistema

#### Verificación
```http
GET /health
```

Devuelve el estado del servidor e información del modelo. **No se requiere autenticación**.

**Respuesta:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu",
  "num_classes": 2286,
  "max_seq_length": 64
}
```

#### Obtener Etiquetas
```http
GET /labels
```

Devuelve todas las etiquetas disponibles.

**Respuesta:**
```json
{
  "num_classes": 2286,
  "labels": {
    "0": "hola",
    "1": "adiós",
    ...
  }
}
```

#### Obtener Configuración
```http
GET /config
```

Devuelve la configuración de inferencia activa para depuración.

### Puntos Finales de Predicción

#### Predecir un Signo Individual
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

**Parámetros:**
- `keypoints`: Array 2D de forma (num_frames, 858)
  - Mano izquierda: 63 características
  - Mano derecha: 63 características
  - Cara: 204 características
  - Cuerpo: 99 características
  - Velocidad: 429 características
- Máximo de fotogramas por solicitud: 512

**Respuesta:**
```json
{
  "prediction": {
    "label": "hola",
    "confidence": 0.9876,
    "top_k": [
      {"label": "hola", "confidence": 0.9876},
      {"label": "hi", "confidence": 0.0098},
      {"label": "saludos", "confidence": 0.0015}
    ],
    "start_frame": 5,
    "end_frame": 42
  }
}
```

#### Procesar Oración
```http
POST /process/sentence
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
  "words": ["hola", "cómo", "estás", "tú"]
}
```

Procesa una lista de palabras a través del LLM para contextualización y creación de oraciones.

**Respuesta:**
```json
{
  "sentence": "¡Hola, cómo estás?"
}
```

## Configuración del Modelo

El modelo se configura a través de `config.yaml`. Parámetros clave:

```yaml
data:
  num_classes: 2286           # Total de signos ASL
  max_seq_length: 64          # Longitud máxima de secuencia
  pose_feature_dim: 858       # Dimensión de características de entrada

model:
  model_type: transformer
  hidden_dim: 512
  num_layers: 4
  num_heads: 8
  ff_dim: 2048
  dropout: 0.2
```

## Autenticación y Limitación de Velocidad

### Autenticación por Clave API

Todos los puntos finales protegidos requieren un encabezado `Authorization`:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:8000/predict/sign
```

### Limitación de Velocidad

- Predeterminado: 30 solicitudes por minuto por clave API
- Configurable a través de variables de entorno

## Ejecutar Pruebas

```bash
# Ejecutar todas las pruebas
pytest

# Ejecutar con cobertura
pytest --cov=app --cov-report=html

# Ver informe de cobertura
open htmlcov/index.html
```

### Variables de Entorno

| Variable | Predeterminado | Descripción |
|----------|---------|-------------|
| `API_KEY` | Requerido | Clave secreta para autenticación de API |
| `GOOGLE_API_KEY` | Requerido | Clave API de Google para características de LLM |
| `ALLOWED_ORIGINS` | Localhost | Orígenes permitidos de CORS (separados por comas) |
| `DEVICE` | `cpu` | Dispositivo de procesamiento (`cpu` o `cuda`) |
| `CHECKPOINT_PATH` | `models/best_model.pt` | Ruta al punto de control del modelo |
| `CONFIG_PATH` | `config.yaml` | Ruta al archivo de configuración |
| `LABELS_PATH` | `models/labels.json` | Ruta al archivo de etiquetas |

## Consideraciones de Rendimiento

- **Tiempo de Inferencia**: ~50-100ms por predicción de seña
- **Solicitudes Concurrentes**: Soporta 2 solicitudes de inferencia concurrentes por defecto
- **Uso de Memoria**: ~2GB típico (entrenado en GPU, se ejecuta en CPU)
- **Máximo de Fotogramas**: 512 fotogramas por solicitud

## Formato de Datos de Entrada

### Extracción de Puntos Clave

Los puntos clave deben extraerse usando la solución Holistic de MediaPipe con la siguiente estructura:

1. **Mano Izquierda**: 21 puntos de referencia × 3 coordenadas = 63 características
2. **Mano Derecha**: 21 puntos de referencia × 3 coordenadas = 63 características
3. **Cara**: 468 puntos de referencia → agrupados en 204 características
4. **Cuerpo**: 33 puntos de referencia × 3 coordenadas = 99 características
5. **Velocidad**: Derivadas temporales de todos los puntos clave = 429 características

Total: 858 características por fotograma

## Manejo de Errores

La API devuelve respuestas de error estandarizadas:

```json
{
  "detail": "Mensaje de error describiendo qué salió mal"
}
```

**Códigos HTTP Comunes:**
- `200 OK`: Solicitud exitosa
- `400 Bad Request`: Datos de entrada inválidos
- `401 Unauthorized`: Clave API faltante o inválida
- `413 Payload Too Large`: La entrada excede el límite máximo de fotogramas
- `422 Unprocessable Entity`: Formato de datos inválido
- `429 Too Many Requests`: Límite de velocidad excedido
- `500 Internal Server Error`: Error del servidor durante la inferencia
- `503 Service Unavailable`: Modelo no cargado todavía