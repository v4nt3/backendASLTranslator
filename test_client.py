
import argparse
import os
import sys
import time
from enum import Enum, auto
from typing import List, Optional

import cv2 #type: ignore
import mediapipe as mp #type: ignore
import numpy as np #type: ignore
import requests #type: ignore

from dotenv import load_dotenv #type: ignore
load_dotenv()


FEATURE_DIM = 858 #dimension
FACE_LANDMARKS_SUBSET = list(range(68))



class PoseExtractor:
    """
    It extracts the same 858 features that the backend expects.

    Structure per frame:
        left_hand  (21 x 3 = 63)
        right_hand (21 x 3 = 63)
        face       (68 x 3 = 204)
        body       (33 x 3 = 99)
        velocity   (429)
    Total: 429 + 429 = 858
    """

    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.5,
        )
        self._prev_keypoints: Optional[np.ndarray] = None

    def reset(self):
        self._prev_keypoints = None

    def extract(self, frame_rgb: np.ndarray):
        """
        Returns:
            features      (858,) float32
            hands_visible bool
        """
        results = self.holistic.process(frame_rgb)
        hands_visible = (
            results.left_hand_landmarks is not None
            or results.right_hand_landmarks is not None
        )
        keypoints = self._build_keypoints(results)

        velocity = (
            keypoints - self._prev_keypoints
            if self._prev_keypoints is not None
            else np.zeros_like(keypoints)
        )
        self._prev_keypoints = keypoints.copy()

        features = np.concatenate([keypoints, velocity]).astype(np.float32)
        assert features.shape[0] == FEATURE_DIM, (
            f"Feature dim mismatch: got {features.shape[0]}, expected {FEATURE_DIM}"
        )
        return features, hands_visible

    def _build_keypoints(self, results) -> np.ndarray:
        kp = []

        # Left hand (63)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                kp.extend([lm.x, lm.y, lm.z])
        else:
            kp.extend([0.0] * 63)

        # Right hand (63)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                kp.extend([lm.x, lm.y, lm.z])
        else:
            kp.extend([0.0] * 63)

        # Face subset (204)
        if results.face_landmarks:
            lms = results.face_landmarks.landmark
            for idx in FACE_LANDMARKS_SUBSET:
                lm = lms[idx] if idx < len(lms) else None
                kp.extend([lm.x, lm.y, lm.z] if lm else [0.0, 0.0, 0.0])
        else:
            kp.extend([0.0] * 204)

        # Body (99)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                kp.extend([lm.x, lm.y, lm.z])
        else:
            kp.extend([0.0] * 99)

        return np.array(kp)

    def close(self):
        self.holistic.close()


class BackendClient:
    def __init__(self, base_url: str, api_key: str):
        self.predict_url = f"{base_url.rstrip('/')}/predict/sign"
        self.health_url  = f"{base_url.rstrip('/')}/health"
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def health(self) -> dict:
        resp = self.session.get(self.health_url, timeout=5)
        resp.raise_for_status()
        return resp.json()

    def predict(self, frames: List[np.ndarray], retries: int = 3, retry_delay: float = 1.0) -> Optional[dict]:

        payload = {"keypoints": [f.tolist() for f in frames]}
        for attempt in range(1, retries + 1):
            try:
                resp = self.session.post(self.predict_url, json=payload, timeout=15)
                if resp.status_code == 503 and attempt < retries:
                    wait = float(resp.headers.get("Retry-After", retry_delay))
                    print(f"[RETRY {attempt}/{retries}] Servidor ocupado, reintentando en {wait:.1f}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.HTTPError:
                print(f"\n[ERROR HTTP {resp.status_code}] {resp.text}")
                return None
            except requests.exceptions.RequestException as e:
                print(f"\n[ERROR de conexión] {e}")
                return None
        print(f"[ERROR] Servidor ocupado luego de {retries} intentos.")
        return None


# state machine
class State(Enum):
    IDLE       = auto()
    SIGNING    = auto()
    PREDICTING = auto()


STATE_COLORS = {
    State.IDLE:       (100, 100, 100),
    State.SIGNING:    (0, 200, 50),
    State.PREDICTING: (0, 200, 255),
}
STATE_LABELS = {
    State.IDLE:       "Waiting hands",
    State.SIGNING:    "Registering signal",
    State.PREDICTING: "Sending to the backend",
}


def draw_ui(
    frame: np.ndarray,
    state: State,
    sentence: List[str],
    last_label: Optional[str],
    last_conf: Optional[float],
    latency_ms: Optional[float],
):
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, h - 140), (w, h), (0, 0, 0), -1)

    # Indicador de estado
    color = STATE_COLORS[state]
    cv2.circle(frame, (25, 25), 12, color, -1)
    cv2.putText(frame, STATE_LABELS[state], (45, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Última predicción
    if last_label:
        lat_str = f" | {latency_ms:.0f} ms" if latency_ms else ""
        cv2.putText(
            frame,
            f"Ultima seña: {last_label} ({last_conf:.1%}){lat_str}",
            (20, h - 110),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
        )

    # Oración acumulada
    sentence_text = " ".join(sentence) or "Esperando señas..."
    cv2.putText(frame, sentence_text, (20, h - 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Instrucciones
    cv2.putText(
        frame,
        "[Enter]: Enviar  [Backspace]: Borrar ultima  [Q]: Salir",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1,
    )


def run(
    base_url: str,
    api_key: str,
    camera_index: int = 0,
    no_hand_patience: int = 10,
    cooldown_frames: int = 15,
    min_frames: int = 10,
    display: bool = True,
):
    client = BackendClient(base_url, api_key)
    print(f"\nConectando a {base_url} ...")
    try:
        health = client.health()
        print(f"[OK] Backend listo: {health}")
    except Exception as e:
        print(f"[ERROR] No se puede conectar al backend: {e}")
        sys.exit(1)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] No se puede abrir la cámara {camera_index}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    max_sign_frames = int(fps * 5)

    extractor = PoseExtractor()

    # Estado
    state              = State.IDLE
    sign_frames: List[np.ndarray] = []
    no_hand_counter    = 0
    cooldown_remaining = 0

    # Oración
    sentence:   List[str]       = []
    last_label: Optional[str]   = None
    last_conf:  Optional[float] = None
    last_lat:   Optional[float] = None

    print("\nCámara iniciada.")
    print("Muestra tus manos → empieza a grabar.")
    print("Baja las manos    → envía al backend.")
    print("[Enter] Enviar oración | [Backspace] Borrar última | [Q] Salir\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame     = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            features, hands_visible = extractor.extract(frame_rgb)

            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                if display:
                    draw_ui(frame, state, sentence, last_label, last_conf, last_lat)
                    cv2.imshow("Test Client — Sign Language", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    _handle_key(key, sentence)
                continue

            if state == State.IDLE:
                if hands_visible:
                    state           = State.SIGNING
                    sign_frames     = [features]
                    no_hand_counter = 0

            elif state == State.SIGNING:
                sign_frames.append(features)
                no_hand_counter = 0 if hands_visible else no_hand_counter + 1

                if (no_hand_counter >= no_hand_patience
                        or len(sign_frames) >= max_sign_frames):
                    state = State.PREDICTING

            elif state == State.PREDICTING:
                if display:
                    draw_ui(frame, state, sentence, last_label, last_conf, last_lat)
                    cv2.imshow("Test Client — Sign Language", frame)
                    cv2.waitKey(1)

                if len(sign_frames) >= min_frames:
                    t0 = time.perf_counter()
                    result = client.predict(sign_frames)
                    latency_ms = (time.perf_counter() - t0) * 1000

                    if result and result.get("success"):
                        pred      = result["prediction"]
                        last_label = pred["label"]
                        last_conf  = pred["confidence"]
                        last_lat   = latency_ms
                        sentence.append(last_label)

                        print(
                            f"[Seña] {last_label:20s} "
                            f"conf={last_conf:.1%}  "
                            f"latencia={latency_ms:.0f}ms"
                        )
                        print(f"       Top-k: {pred['top_k'][:3]}")
                        print(f"       Oración: {' '.join(sentence)}\n")
                    else:
                        print("[WARN] Predicción descartada (sin resultado del backend)")
                else:
                    print(
                        f"[SKIP] Seña muy corta ({len(sign_frames)} frames "
                        f"< {min_frames} mínimo)"
                    )

                sign_frames        = []
                no_hand_counter    = 0
                cooldown_remaining = cooldown_frames
                state              = State.IDLE

            if display:
                draw_ui(frame, state, sentence, last_label, last_conf, last_lat)
                cv2.imshow("Test Client — Sign Language", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                _handle_key(key, sentence)

    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        extractor.close()
        print("\nSesión terminada.")
        if sentence:
            print(f"Última oración: {' '.join(sentence)}")


def _handle_key(key: int, sentence: List[str]):
    if key in (13, 10):   # Enter
        oracion = " ".join(sentence)
        print(f"\n[ORACIÓN ENVIADA]: {oracion}\n")
        sentence.clear()
    elif key in (8, 127): # Backspace
        if sentence:
            print(f"[Borrado]: {sentence.pop()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulador de frontend para pruebas locales del backend de lengua de señas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",        default="http://localhost:8000",
        help="URL base del backend FastAPI",
    )
    parser.add_argument(
        "--api-key",    default=os.getenv("API_KEY", ""),
        help="Valor del header X-API-Key (o variable de entorno API_KEY)",
    )
    parser.add_argument("--camera",     type=int,   default=0,  help="Índice de cámara")
    parser.add_argument("--patience",   type=int,   default=10, help="Frames sin manos para cerrar una seña")
    parser.add_argument("--cooldown",   type=int,   default=15, help="Frames de pausa post-predicción")
    parser.add_argument("--min-frames", type=int,   default=10, help="Mínimo de frames para enviar al backend")
    parser.add_argument("--no-display", action="store_true",    help="Desactivar ventana de OpenCV")

    args = parser.parse_args()

    if not args.api_key:
        print("[ERROR] Falta la API key.")
        print("        Pásala con --api-key <key> o define la variable de entorno API_KEY.")
        sys.exit(1)

    run(
        base_url        = args.url,
        api_key         = args.api_key,
        camera_index    = args.camera,
        no_hand_patience= args.patience,
        cooldown_frames = args.cooldown,
        min_frames      = args.min_frames,
        display         = not args.no_display,
    )