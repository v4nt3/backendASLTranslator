import asyncio
import time
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
from fastapi import HTTPException


# SECURITY.PY — RateLimiter

class TestRateLimiter:

    def _make_request(self, ip: str = "127.0.0.1") -> MagicMock:
        request = MagicMock()
        request.client.host = ip
        request.headers.get = MagicMock(return_value=None)  # sin X-Forwarded-For
        return request

    def test_primer_request_es_permitido(self):
        from app.security import RateLimiter
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        request = self._make_request()
        result = limiter.check(request)
        assert isinstance(result, dict)
        assert "X-RateLimit-Remaining" in result

    def test_requests_dentro_del_limite_son_permitidos(self):
        from app.security import RateLimiter
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        request = self._make_request()
        for _ in range(5):
            limiter.check(request)  # no debe lanzar excepción

    def test_bloquea_al_superar_limite(self):
        from app.security import RateLimiter
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        request = self._make_request()

        for _ in range(3):
            limiter.check(request)

        with pytest.raises(HTTPException) as exc_info:
            limiter.check(request)

        assert exc_info.value.status_code == 429

    def test_respuesta_429_incluye_retry_after(self):
        from app.security import RateLimiter
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        request = self._make_request()

        for _ in range(2):
            limiter.check(request)

        with pytest.raises(HTTPException) as exc_info:
            limiter.check(request)

        assert "Retry-After" in exc_info.value.headers
        assert int(exc_info.value.headers["Retry-After"]) > 0

    def test_ips_distintas_tienen_contadores_independientes(self):
        from app.security import RateLimiter
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        req_a = self._make_request("192.168.1.1")
        req_b = self._make_request("192.168.1.2")

        # Agota el límite de IP A
        for _ in range(2):
            limiter.check(req_a)

        # IP B todavía puede hacer requests
        limiter.check(req_b)  # no debe lanzar excepción

        # IP A ya no puede
        with pytest.raises(HTTPException) as exc_info:
            limiter.check(req_a)
        assert exc_info.value.status_code == 429

    def test_requests_antiguos_expiran_fuera_de_ventana(self):
        from app.security import RateLimiter
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        request = self._make_request()

        # Inserta timestamps artificialmente viejos (fuera de la ventana)
        ip = "127.0.0.1"
        old_time = time.time() - 120  # 120s atrás, ventana es 60s
        limiter._requests[ip] = [old_time, old_time]

        # Ahora el request debe pasar porque los anteriores expiraron
        result = limiter.check(request)
        assert isinstance(result, dict)

    def test_headers_de_rate_limit_son_correctos(self):
        from app.security import RateLimiter
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        request = self._make_request()

        result = limiter.check(request)

        assert result["X-RateLimit-Limit"] == "5"
        assert result["X-RateLimit-Remaining"] == "4"  # 5 - 0 anteriores - 1 actual
        assert "X-RateLimit-Reset" in result

    def test_remaining_decrece_con_cada_request(self):
        from app.security import RateLimiter
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        request = self._make_request()

        r1 = limiter.check(request)
        r2 = limiter.check(request)
        r3 = limiter.check(request)

        assert int(r1["X-RateLimit-Remaining"]) == 4
        assert int(r2["X-RateLimit-Remaining"]) == 3
        assert int(r3["X-RateLimit-Remaining"]) == 2

    def test_cleanup_elimina_ip_sin_requests_activos(self):
        from app.security import RateLimiter
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        ip = "10.0.0.1"

        # Inserta un timestamp viejo
        limiter._requests[ip] = [time.time() - 120]

        # Trigger cleanup
        limiter._cleanup(ip, time.time())

        # La IP debe haber sido eliminada del dict
        assert ip not in limiter._requests

# SECURITY.PY — verify_api_key

class TestVerifyApiKey:

    @pytest.mark.asyncio
    async def test_acepta_key_correcta(self):
        from app.security import verify_api_key
        with patch("app.security.get_api_key", return_value="mi-key-secreta"):
            result = await verify_api_key(api_key="mi-key-secreta")
            assert result == "mi-key-secreta"

    @pytest.mark.asyncio
    async def test_rechaza_sin_key(self):
        from app.security import verify_api_key
        with patch("app.security.get_api_key", return_value="mi-key-secreta"):
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(api_key=None)
            assert exc_info.value.status_code == 403
            assert "Missing API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_rechaza_key_incorrecta(self):
        from app.security import verify_api_key
        with patch("app.security.get_api_key", return_value="mi-key-secreta"):
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(api_key="key-incorrecta")
            assert exc_info.value.status_code == 403
            assert "Invalid API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_devuelve_503_si_no_hay_key_configurada(self):
        from app.security import verify_api_key
        with patch("app.security.get_api_key", return_value=None):
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(api_key="cualquier-key")
            assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_rechaza_key_vacia(self):
        from app.security import verify_api_key
        with patch("app.security.get_api_key", return_value="mi-key-secreta"):
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(api_key="")
            assert exc_info.value.status_code == 403

# MAIN.PY — _validate_keypoints

class TestValidateKeypoints:

    def _valid_frame(self, n=1):
        return [[0.0] * 858 for _ in range(n)]

    def test_input_valido_retorna_array_numpy(self):
        from app.main import _validate_keypoints
        result = _validate_keypoints(self._valid_frame(20))
        assert isinstance(result, np.ndarray)
        assert result.shape == (20, 858)
        assert result.dtype == np.float32

    def test_un_frame_valido(self):
        from app.main import _validate_keypoints
        result = _validate_keypoints(self._valid_frame(1))
        assert result.shape == (1, 858)

    def test_exactamente_512_frames_es_valido(self):
        from app.main import _validate_keypoints
        result = _validate_keypoints(self._valid_frame(512))
        assert result.shape == (512, 858)

    def test_array_vacio_lanza_422(self):
        from app.main import _validate_keypoints
        with pytest.raises(HTTPException) as exc_info:
            _validate_keypoints([])
        assert exc_info.value.status_code == 422
        assert "empty" in exc_info.value.detail

    def test_dimension_incorrecta_lanza_422(self):
        from app.main import _validate_keypoints
        frames_mal = [[0.0] * 100 for _ in range(5)]  # 100 en vez de 858
        with pytest.raises(HTTPException) as exc_info:
            _validate_keypoints(frames_mal)
        assert exc_info.value.status_code == 422
        assert "858" in exc_info.value.detail

    def test_513_frames_lanza_413(self):
        from app.main import _validate_keypoints
        with pytest.raises(HTTPException) as exc_info:
            _validate_keypoints(self._valid_frame(513))
        assert exc_info.value.status_code == 413
        assert "512" in exc_info.value.detail

    def test_mensaje_413_menciona_cantidad_recibida(self):
        from app.main import _validate_keypoints
        with pytest.raises(HTTPException) as exc_info:
            _validate_keypoints(self._valid_frame(600))
        assert "600" in exc_info.value.detail

    def test_array_1d_lanza_422(self):
        from app.main import _validate_keypoints
        with pytest.raises(HTTPException) as exc_info:
            _validate_keypoints([0.0, 0.1, 0.2])
        assert exc_info.value.status_code == 422

    def test_valores_float_son_preservados(self):
        from app.main import _validate_keypoints
        frame = [float(i) / 858 for i in range(858)]
        result = _validate_keypoints([frame])
        assert result.shape == (1, 858)
        assert abs(result[0][0] - 0.0) < 1e-5

    def test_valores_negativos_son_validos(self):
        from app.main import _validate_keypoints
        frame = [-0.5] * 858
        result = _validate_keypoints([frame])
        assert result.shape == (1, 858)

# LLM_SERVICE.PY — process_sentence

class TestProcessSentence:

    def test_retorna_string_cuando_api_funciona(self):
        from app.llm_service import process_sentence

        mock_response = MagicMock()
        mock_response.text = "What time is it?"

        with patch("app.llm_service.model") as mock_model:
            mock_model.generate_content.return_value = mock_response
            result = process_sentence("TIME WHAT")

        assert isinstance(result, str)
        assert result == "What time is it?"

    def test_retorna_input_original_si_api_falla(self):
        from app.llm_service import process_sentence

        with patch("app.llm_service.model") as mock_model:
            mock_model.generate_content.side_effect = Exception("API error")
            result = process_sentence("TIME WHAT")

        assert result == "TIME WHAT"

    def test_strip_en_respuesta(self):
        from app.llm_service import process_sentence

        mock_response = MagicMock()
        mock_response.text = "  What time is it?  \n"

        with patch("app.llm_service.model") as mock_model:
            mock_model.generate_content.return_value = mock_response
            result = process_sentence("TIME WHAT")

        assert result == "What time is it?"
        assert not result.startswith(" ")
        assert not result.endswith("\n")

    def test_fallback_con_error_de_quota(self):
        from app.llm_service import process_sentence

        with patch("app.llm_service.model") as mock_model:
            mock_model.generate_content.side_effect = Exception(
                "429 Resource exhausted"
            )
            result = process_sentence("RAIN HERE")

        assert result == "RAIN HERE"

    def test_llama_a_generate_content_con_el_prompt_correcto(self):
        from app.llm_service import process_sentence

        mock_response = MagicMock()
        mock_response.text = "It is raining here."

        with patch("app.llm_service.model") as mock_model:
            mock_model.generate_content.return_value = mock_response
            process_sentence("RAIN HERE")

            call_args = mock_model.generate_content.call_args[0][0]
            assert "RAIN HERE" in call_args
            assert "ASL" in call_args

    def test_string_vacio_retorna_string_vacio(self):
        from app.llm_service import process_sentence

        with patch("app.llm_service.model") as mock_model:
            mock_model.generate_content.side_effect = Exception("error")
            result = process_sentence("")

        assert result == ""
        assert isinstance(result, str)

    def test_no_retorna_none(self):
        from app.llm_service import process_sentence

        with patch("app.llm_service.model") as mock_model:
            mock_model.generate_content.side_effect = Exception("error")
            result = process_sentence("ANYTHING")

        assert result is not None