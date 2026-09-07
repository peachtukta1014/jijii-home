#!/bin/sh
# Jijii Home entrypoint for Cloud Run.
# Cloud Run gives $PORT; wire it to the proxy's env. Host 0.0.0.0 (in-container).
export TTS_PROXY_HOST="${TTS_PROXY_HOST:-0.0.0.0}"
export TTS_PROXY_PORT="${PORT:-8080}"
echo "[jijii-home] starting proxy on 0.0.0.0:$TTS_PROXY_PORT"
exec python3 tts_openrouter_proxy.py