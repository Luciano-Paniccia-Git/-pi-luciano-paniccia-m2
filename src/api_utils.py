"""
api_utils.py — Wrapper con reintentos para llamadas a la API de OpenAI.

En el proyecto del módulo anterior, el feedback marcó como pendiente el manejo
de errores de red y rate-limit (429, 5xx): una falla de la API hacía explotar
todo el programa con un traceback. Acá lo resolvemos con un decorador simple
de reintento con backoff exponencial, para que un rate-limit o un error
transitorio del servidor no tire abajo todo el pipeline.
"""

from __future__ import annotations

import time
import functools


def with_retries(max_retries: int = 3, base_delay_seconds: float = 1.5):
    """
    Decorador que reintenta una llamada a la API de OpenAI ante errores
    transitorios (rate limit, timeouts, errores 5xx del servidor), con
    backoff exponencial (1.5s, 3s, 6s...). Errores que NO son transitorios
    (ej. API key inválida, request mal formado) se relanzan de inmediato,
    sin reintentar, porque reintentar no los va a solucionar.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from openai import (
                RateLimitError,
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
            )

            transient_errors = (
                RateLimitError,
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
            )

            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except transient_errors as e:
                    last_error = e
                    if attempt == max_retries:
                        break
                    delay = base_delay_seconds * (2**attempt)
                    print(
                        f"[api_utils] Error transitorio ({type(e).__name__}), "
                        f"reintentando en {delay:.1f}s (intento {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(delay)
            raise last_error

        return wrapper

    return decorator
