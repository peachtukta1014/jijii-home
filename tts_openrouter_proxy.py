#!/usr/bin/env python3
"""
tts_openrouter_proxy.py — Local TTS proxy for the Minis app (iSH on iPhone).

WHY THIS EXISTS (Peach's approach, same as Marin):
The app's "Voice Output" group routes through a provider whose customBaseURL
points here (http://localhost:8765). The app sends OpenAI-style
POST /audio/speech; this proxy rewrites the request per-model and forwards to
OpenRouter's TTS API, then returns a playable mp3. The app itself does NOT need
its own voice picker or SDK — it just talks to this proxy endpoint, exactly like
the OpenAI/Marin setup we already had.

Models served (Thai, female voice) — fish was dropped (free tier only had a
default/alloy voice; no female Thai ID we could reliably set):
  1. qwen/qwen-audio-3.0-tts-flash       -> voice=longanhuan_v3.6 (female), mp3
  2. google/gemini-3.1-flash-tts-preview -> voice=Kore (female Thai, reads
                                            mixed Thai+English fully),
                                            response_format=pcm (Gemini ONLY
                                            accepts pcm), converted to mp3
                                            locally via ffmpeg so the app always
                                            plays an mp3.

Auth: proxy uses $OAUTH_OPENROUTER_KEY (the key that works for OpenRouter TTS;
the OpenRouter Management key returns 401 for TTS). Client auth is ignored.

Forwarded paths (passthrough for non-TTS):
  POST /audio/speech   -> OpenRouter TTS (rewritten per model)
  POST /chat/completions, /responses, /audio/transcriptions -> passthrough
  GET  /models         -> passthrough

Run:  python3 tts_openrouter_proxy.py
"""

import base64
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("TTS_PROXY_PORT", "8765"))
# Host to bind. Default 127.0.0.1 for local iSH. On Cloud Run set TTS_PROXY_HOST=0.0.0.0
# so the container answers on the $PORT Cloud Run gives us (0.0.0.0 inside).
HOST = os.environ.get("TTS_PROXY_HOST", "127.0.0.1")
UPSTREAM = os.environ.get("TTS_UPSTREAM", "https://openrouter.ai/api")
# The key that actually works for OpenRouter TTS.
API_KEY = os.environ.get("OAUTH_OPENROUTER_KEY", "")
# Peach 3/9: Gemini direct (OpenRouter no longer lists Gemini TTS) — when
# TTS_GEMINI_DIRECT=1, gemini requests go straight to generativelanguage
# with $GEMINI_API_KEY instead of through OpenRouter.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TTS_GEMINI_DIRECT = os.environ.get("TTS_GEMINI_DIRECT", "0") == "1"

# Female-Thai voices from the 2026-08-27 OpenRouter TTS guide.
QWEN_VOICE = os.environ.get("QWEN_VOICE", "longanhuan_v3.6")
# Peach 2026-08-30: qwen is the DEFAULT/main voice — model selectable via env
# (set in tts_model.conf by select-tts-model.sh / start_tts_proxy.sh).
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen/qwen-audio-3.0-tts-flash")
# Peach 2026-08-30: MAI (Azure Neural Thai female) is the secondary/backup
# voice (reads Thai clearly, less Chinese leakage than qwen).
MAI_MODEL = os.environ.get("MAI_MODEL", "microsoft/mai-voice-2-flash")
# Peach 2026-08-28: use Kore (Gemini female, soft) for English/mixed reads —
# qwen is locked to Thai (reads Thai only, skips English). Kore reads both.
GEMINI_VOICE = os.environ.get("GEMINI_VOICE", "Kore")
# Peach 2026-08-29 (UNIFIED per-model voice env): each TTS AI model has its OWN
# env variable for its Thai-female voice, exactly like GEMINI_VOICE / QWEN_VOICE,
# so calling a model always pairs it with that model's voice env. Defaults are
# the verified Thai-female voices (fish = Cyn Thai female; minimax = tender woman;
# azure/mai = Premwadee Neural). Override any per model via its env.
FISH_VOICE = os.environ.get("FISH_VOICE", "4fb6a62fc543463c95bd5cd5e3fdb269")  # เสียงจีจี้ (Fish voice model, 5/9)
FISH_FREE_VOICE = os.environ.get("FISH_FREE_VOICE", "4fb6a62fc543463c95bd5cd5e3fdb269")
# 4/9: fish API key — peach's fish key lives in AZURE_API_KEY (verified GET /model 200).
# If FISH_API_KEY is set but looks wrong (not a 51-char sk-...), fall back to AZURE_API_KEY.
FISH_API_KEY = os.environ.get("FISH_API_KEY", os.environ.get("AZURE_API_KEY", ""))
if not (FISH_API_KEY.startswith("sk-") and len(FISH_API_KEY) >= 40):
    FISH_API_KEY = os.environ.get("AZURE_API_KEY", "")
FISH_API_MODEL = os.environ.get("FISH_API_MODEL", "s2.1-pro-free")  # free developer tier (no API credit needed)

# (Peach 2026-09-05: removed fish 1M/mo quota guard — that tier belongs to
# Google Cloud TTS, NOT fish. fish quota is tracked in quota_state.json for
# observability only; no hard limit enforced here.)
FISH_LATENCY = os.environ.get("FISH_LATENCY", "balanced")        # fish latency: normal|balanced|fast
FISH_TIMEOUT = int(os.environ.get("TTS_FISH_TIMEOUT", "90"))     # generous: long Thai text
QUOTA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quota_state.json")

# --- fish.audio VOICE-DESIGN knobs (2026-09-05 audit — Peach: "เสียงเพี้ยน/พูดไม่ชัด") ---
# Official /v1/tts schema: docs.fish.audio/api-reference/endpoint/openapi-v1/text-to-speech.md
# Audit found the proxy previously sent ONLY {text, reference_id, format, latency} —
# fish DEFAULTS (temperature=0.7, top_p=0.7, repetition_penalty=1.2, chunk_length=300,
# normalize=true) are too RANDOM for clear Thai. We now pin stable values (env-overridable)
# so pronunciation stays consistent read-to-read:
#   temperature 0.4  -> deterministic, fewer pronunciation slips
#   repetition_penalty 1.3 -> >=1.0 reduces syllable/repetition loops
#   prosody.speed 0.95 -> unhurried pace (0.5-2.0 allowed) => talk clearly
#   normalize true -> official EN/CN text normalization
#   FISH_NORMALIZE_TH=1 -> OUR Thai normalizer (numbers/currency/phones -> Thai words)
FISH_TEMPERATURE = float(os.environ.get("FISH_TEMPERATURE", "0.4"))
FISH_TOP_P = float(os.environ.get("FISH_TOP_P", "0.7"))
FISH_REPETITION_PENALTY = float(os.environ.get("FISH_REPETITION_PENALTY", "1.3"))
FISH_SPEED = float(os.environ.get("FISH_SPEED", "0.95"))
FISH_CHUNK_LENGTH = int(os.environ.get("FISH_CHUNK_LENGTH", "300"))
FISH_NORMALIZE = os.environ.get("FISH_NORMALIZE", "1") == "1"
FISH_QUALITY_GUARD = os.environ.get("FISH_QUALITY_GUARD", "0") == "1"   # features:["quality-guard"]
FISH_NORMALIZE_TH = os.environ.get("FISH_NORMALIZE_TH", "1") == "1"     # Thai read-aloud normalizer

# --- Google Cloud TTS monthly quota guard (Peach 6/9: GCP free tier = 1M chars/mo;
# fish audio has its OWN quota — NOT limited here). REBUILD-2: prevents the
# 515฿ overage bill. Guard applies ONLY to the gcp route (cloud_chars). ---
CLOUD_MONTHLY_LIMIT = int(os.environ.get("CLOUD_MONTHLY_LIMIT", "1000000"))
CLOUD_WARN_PCT = float(os.environ.get("CLOUD_WARN_PCT", "80"))

# --- Pronunciation control dictionary (Peach 6/9) ---
# Custom word/phrase pronunciations the app owner defines (names, shop terms,
# loanwords...). JSON file next to this proxy: {"phrase": "how-to-read"}.
# Longest phrase wins; applied at the START of Thai normalization, so the model
# reads the override instead of guessing. Env TTS_PRONUNCIATION_FILE overrides.
PRONUNCIATION_FILE = os.environ.get(
    "TTS_PRONUNCIATION_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "pronunciation_dict.json"))
_PRONUNCIATION_DICT = None


def _load_pronunciation_dict() -> dict:
    """Load the pronunciation override dictionary (cached). Never crashes."""
    global _PRONUNCIATION_DICT
    if _PRONUNCIATION_DICT is None:
        try:
            with open(PRONUNCIATION_FILE, encoding="utf-8") as f:
                d = json.load(f)
            _PRONUNCIATION_DICT = {str(k).strip(): str(v).strip()
                                   for k, v in d.items() if str(k).strip()}
        except Exception:
            _PRONUNCIATION_DICT = {}
    return _PRONUNCIATION_DICT


def _apply_pronunciation_dict(text: str) -> str:
    """Replace custom pronounced phrases (longest first). Case-insensitive for
    latin; exact for Thai (no case). E.g. {'จีจี้': 'จีจี้'} is a no-op by
    design — use it for words the model misreads (loanwords, names, jargon)."""
    d = _load_pronunciation_dict()
    if not d or not text:
        return text
    for phrase in sorted(d, key=len, reverse=True):
        if not phrase or phrase.startswith("_") or not d.get(phrase):
            continue
        repl = d[phrase]
        if phrase == repl:
            continue
        if re.search(r"[A-Za-z]", phrase):
            text = re.sub(re.escape(phrase), repl, text, flags=re.IGNORECASE)
        else:
            text = text.replace(phrase, repl)
    return text

_TH_DIGITS = ["", "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า"]


def _thai_read_int(n: int) -> str:
    """Integer 0..999,999,999,999 -> Thai words (1500 -> หนึ่งพันห้าร้อย)."""
    if n == 0:
        return "ศูนย์"
    neg = "ลบ" if n < 0 else ""
    n = abs(n)
    if n >= 10 ** 12:
        return neg + str(n)  # beyond ล้านล้าน — leave digits (rare in speech text)
    parts = []
    if n >= 10 ** 6:
        parts.append(_thai_read_int(n // 10 ** 6) + "ล้าน")
        parts.append(_thai_read_group6(n % 10 ** 6, True))
    else:
        parts.append(_thai_read_group6(n, False))
    return neg + "".join(parts)


def _thai_read_group6(n: int, has_higher: bool) -> str:
    """0 < n < 1,000,000 -> Thai words. has_higher = a ล้าน group precedes."""
    if n == 0:
        return ""
    out = []
    s = n // 100000
    if s: out.append(_TH_DIGITS[s] + "แสน")
    m = (n // 10000) % 10
    if m: out.append(_TH_DIGITS[m] + "หมื่น")
    k = (n // 1000) % 10
    if k: out.append(_TH_DIGITS[k] + "พัน")
    h = (n // 100) % 10
    if h: out.append("หนึ่งร้อย" if h == 1 else _TH_DIGITS[h] + "ร้อย")
    t = (n // 10) % 10
    u = n % 10
    if t == 1: out.append("สิบ")
    elif t == 2: out.append("ยี่สิบ")
    elif t: out.append(_TH_DIGITS[t] + "สิบ")
    if u:
        if u == 1 and (has_higher or s or m or k or h or t):
            out.append("เอ็ด")
        else:
            out.append(_TH_DIGITS[u])
    return "".join(out)


def _thai_read_digits(s: str) -> str:
    """Digit string -> Thai digit names (phone/ID reading, e.g. 0891234567)."""
    names = ["ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า"]
    return " ".join(names[int(c)] for c in s if c.isdigit())


def _thai_read_decimal(s: str) -> str:
    """'1,500' / '1.5' / '250' -> Thai words (1.5 -> หนึ่งจุดห้า)."""
    s = s.replace(",", "")
    if "." in s:
        ip, fp = s.split(".", 1)
        if ip:
            return _thai_read_int(int(ip)) + "จุด" + _thai_read_digits(fp)
        return "ศูนย์จุด" + _thai_read_digits(fp)
    return _thai_read_int(int(s))


def _thai_normalize_text(text: str) -> str:
    """Light Thai read-aloud normalizer for fish (raw 1,500 / ฿ / 0xx… reads
    distorted by the model). Converts SAFE patterns only; everything else passes
    through untouched:
      - comma-group integers (1,500) & plain integers 1-6 digits -> Thai words
      - 7+ digit strings (phones/IDs) -> digit-by-digit Thai names
      - current-era years 2400-2599 -> digit-by-digit (2567 -> สอง ห้า หก เจ็ด)
      - decimals (1.5 -> หนึ่งจุดห้า), percents (50% -> ห้าสิบเปอร์เซ็นต์)
      - ฿ -> บาท, $ -> ดอลลาร์; strips emoji/control chars; guarantees a pause."""
    if not text:
        return text
    t = _apply_pronunciation_dict(text)
    t = re.sub(r"[\U0001F000-\U0001FAFF\uFE0F\u200D\u200B\u2060]", "", t)
    t = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", t)
    # currency + amount together (฿250 / $1,500.50) — read the amount in Thai
    t = re.sub(r"฿\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
               lambda m: "บาท" + _thai_read_decimal(m.group(1)), t)
    t = re.sub(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
               lambda m: "ดอลลาร์" + _thai_read_decimal(m.group(1)), t)
    t = t.replace("฿", "บาท").replace("$", "ดอลลาร์")
    N = r"(?<![\d\u0E00-\u0E7F])"   # not preceded by digit/Thai char
    K = r"(?![\d\u0E00-\u0E7F])"    # not followed by digit/Thai char
    # phones/IDs (7+ digits, may follow a Thai word like "เบอร์") -> digit names
    t = re.sub(r"(?<![\d])[0-9]{7,}(?![\d])", lambda m: _thai_read_digits(m.group(0)), t)
    t = re.sub(N + r"[12][0-9]{3}" + K, lambda m: _thai_read_digits(m.group(0)), t)
    t = re.sub(N + r"[0-9]{1,3}(?:,[0-9]{3})+" + K,
               lambda m: _thai_read_int(int(m.group(0).replace(",", ""))), t)
    t = re.sub(N + r"[0-9]{1,6}\.[0-9]{1,3}" + K,
               lambda m: _thai_read_int(int(m.group(0).split(".")[0]))
                         + "จุด" + _thai_read_digits(m.group(0).split(".")[1]), t)
    t = re.sub(N + r"[0-9]{1,3}%" + K,
               lambda m: _thai_read_int(int(m.group(0).rstrip("%"))) + "เปอร์เซ็นต์", t)
    t = re.sub(N + r"[0-9]{1,6}" + K, lambda m: _thai_read_int(int(m.group(0))), t)
    t = re.sub(r"[ \t]+", " ", t).strip()
    if t and t[-1] not in ".!?…":
        t += "."   # force a sentence-ending pause so the model stops cleanly
    return t


def _thai_split_sentences(text: str, maxlen: int = 300, minlen: int = 20) -> list:
    """Split for fish WS streaming at breath boundaries (.!?… space newline),
    keeping chunks <= maxlen and >= minlen where possible — the old fixed
    60-char cut split Thai words mid-word and caused slurred reading."""
    if not text:
        return [""]
    chunks, cur = [], ""
    for ch in text:
        cur += ch
        hard = len(cur) >= maxlen
        soft = (ch in ".!?…\n \t") and len(cur) >= minlen
        if hard or soft:
            chunks.append(cur)
            cur = ""
    if cur:
        chunks.append(cur)
    return chunks or [""]


def _fish_count_chars(n: int):
    """Add n chars to this month's fish usage (quota_state.json). Observability only."""
    if n <= 0:
        return
    try:
        cm = time.strftime("%Y-%m")
        state = {"cmonth": cm, "fish_chars": 0}
        try:
            with open(QUOTA_FILE) as f:
                old = json.load(f)
            if old.get("cmonth") == cm:
                state = old
        except Exception:
            pass
        state["fish_chars"] = state.get("fish_chars", 0) + n
        with open(QUOTA_FILE, "w") as f:
            json.dump(state, f, indent=1)
    except Exception:
        pass  # counting must never break synthesis


def _gcp_quota_state() -> dict:
    """Current month cloud chars used (quota_state.json)."""
    try:
        cm = time.strftime("%Y-%m")
        with open(QUOTA_FILE) as f:
            state = json.load(f)
        if state.get("cmonth") == cm:
            return state
        return {"cmonth": cm, "cloud_chars": 0}
    except Exception:
        return {"cmonth": time.strftime("%Y-%m"), "cloud_chars": 0}


def _gcp_quota_check(n: int) -> tuple:
    """Return (ok, used, limit, pct). Blocks when used+n would exceed limit."""
    st = _gcp_quota_state()
    used = st.get("cloud_chars", 0)
    limit = CLOUD_MONTHLY_LIMIT
    if n + used >= limit:
        return False, used, limit, round(100.0 * (n + used) / limit, 1)
    return True, used, limit, round(100.0 * (n + used) / limit, 1)


def _gcp_count_chars(n: int):
    """Add n chars to this month's cloud usage (only on successful synth)."""
    if n <= 0:
        return
    try:
        st = _gcp_quota_state()
        st["cloud_chars"] = st.get("cloud_chars", 0) + n
        with open(QUOTA_FILE, "w") as f:
            json.dump(st, f, indent=1)
    except Exception:
        pass
MINIMAX_VOICE = os.environ.get("MINIMAX_VOICE", "Thai_tender_woman")
MAI_VOICE = os.environ.get("MAI_VOICE", "th-TH-Premwadee:MAI-Voice-2")
# Peach 2026-08-29: TTS model selector — the Thai main voice model comes from
# env (select-tts-model.sh writes tts_model.conf -> exports GEMINI_MODEL).
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "google/gemini-3.1-flash-tts-preview")

# Language handling for Qwen (Peach 2026-08-30 AGGRESSIVE): qwen reads PURE
# Thai only — latin/numbers/symbols are stripped (QWEN_AGGRESSIVE_STRIP below)
# and language_type is hard-locked to "Thai" in _qwen_payload. QWEN_LANGUAGE
# env is kept for compat but "auto" is never honoured anymore (Thai always).
QWEN_LANGUAGE = os.environ.get("QWEN_LANGUAGE", "Thai")

# Default speaking style for Gemini (override via GEMINI_STYLE env, or per-request "instructions").
# Speaking style for Gemini (override via GEMINI_STYLE env, or per-request "instructions").
# Updated 2026-08-31 per Peach: used across Open Minis app reads (not the shop
# POS anymore) -> read text plainly and naturally as written, no day-close /
# total-expansion theatrics. Keep the Jijii persona: warm Thai female voice.
DEFAULT_GEMINI_STYLE = os.environ.get(
    "GEMINI_STYLE",
    "# AUDIO PROFILE: Jijii (จีจี้) — a warm, bright Thai woman, "
    "a close friend / younger-sister vibe.\n"
    "### DIRECTOR'S NOTES\n"
    "Style: warm, cheerful, natural — a genuine 'vocal smile'. Feminine, "
    "soft, confident, never robotic, never monotone.\n"
    "Pace: relaxed, natural rhythm, easy to follow.\n"
    "Language: Thai is the MAIN language. Read Thai words in Thai and English "
    "words in a natural English accent. Read the text plainly and naturally "
    "exactly as written — no extra expansions, no theatrical emphasis.")

SUPPORTED_FRAGMENTS = (
    # 4/9: qwen/minimax/mai REMOVED — qwen = Chinese model (เสียงจีนปน), and
    # fish is now the primary free Thai-female voice. Keep only what works.
    "gemini-2.5-flash-preview-tts",
    "gemini-3.1-flash-tts-preview",
    "fish-audio/s2.1-pro",          # covers s2.1-pro AND s2.1-pro-free:free (substring)
    # Peach 3/9: Google Cloud TTS (NOT OpenRouter) — th-TH-Neural2-C/A female Thai,
    # synthesized DIRECT via service account (FIREBASE_SERVICE_ACCOUNT env) at
    # texttospeech.googleapis.com. Free tier 1M chars/month.
    "th-TH-Neural2",
)

# Voice Lab (:8766) model catalog — returned by GET /models on the lab port.
# OpenRouter's plain /models does NOT list TTS models (they're only discoverable
# via ?output_modalities=speech), which is why the app marks them "no longer
# available". Serving this catalog on the lab port makes every lab model visible.
LAB_CATALOG = [
    # 4/9: qwen/minimax/mai removed from catalog — only models that actually
    # serve Thai-female correctly stay visible to the app.
    {"id": "fish-audio/s2.1-pro",                   "object": "model", "created": 1754000003,
     "name": "Fish Audio S2.1 Pro (voice cloning)",
     "architecture": {"input_modalities": ["text"], "output_modalities": ["speech"]}},
    {"id": "google/gemini-3.1-flash-tts-preview",   "object": "model", "created": 1754000006,
     "name": "Gemini 3.1 Flash TTS (Sulafat, Thai+EN)",
     "architecture": {"input_modalities": ["text"], "output_modalities": ["speech"]}},
    {"id": "google/gemini-2.5-flash-preview-tts",  "object": "model", "created": 1754000008,
     "name": "Gemini 2.5 Flash TTS (Sulafat, Thai+EN) — ถูกกว่า ครึ่งราคา",
     "architecture": {"input_modalities": ["text"], "output_modalities": ["speech"]}},
    {"id": "fish-audio/s2.1-pro-free:free",         "object": "model", "created": 1754000007,
     "name": "Fish Audio S2.1 Pro Free (เสียงใสพรีเซ็นเตอร์ ของพี่ — ไทย)",
     "architecture": {"input_modalities": ["text"], "output_modalities": ["speech"]}},
    {"id": "th-TH-Neural2-C",                       "object": "model", "created": 1754000009,
     "name": "Google Cloud TTS Thai Female (th-TH-Neural2-C)",
     "architecture": {"input_modalities": ["text"], "output_modalities": ["speech"]}},
]

# Peach 2026-08-28: qwen reads Thai well but, when text mixes in any English
# (latin) word, it switches to English and DROPS all remaining Thai (model
# limitation; Thai full support still rolling out per Qwen-Audio-3.0-TTS blog).
# Deterministic fix (verified): when the text is Thai-primary with stray English
# words, we STRIP the latin words before sending to qwen so it reads pure Thai
# (English words are skipped). Only applies to mixed text; pure-English text is
# sent as-is. Toggle off (0) to send original text untouched.
QWEN_SKIP_LATIN = os.environ.get("TTS_QWEN_SKIP_LATIN", "1") == "1"
# Peach 2026-08-30 (AGGRESSIVE Thai-only): strip ALL non-Thai characters (latin
# letters, digits, punctuation, symbols, emoji) leaving ONLY Thai chars
# (U+0E00-U+0E7F incl. tone marks) + whitespace before sending to qwen, and
# hard-lock language_type="Thai". TTS_QWEN_AGGRESSIVE_STRIP=1 is the default;
# set 0 to keep the classic latin-word-only strip.
QWEN_AGGRESSIVE_STRIP = os.environ.get("TTS_QWEN_AGGRESSIVE_STRIP", "1") == "1"

# --- Error-aware automatic fallback (Peach 2026-08-29) ---
# If the primary model fails with a retryable status (rate limit, provider
# overload, edge timeout, 5xx), the proxy falls back ONCE to the partner model
# so the user never sees a raw error. Toggle off entirely with TTS_FALLBACK=0.
FALLBACK_ENABLED = os.environ.get("TTS_FALLBACK", "0") == "1"  # 4/9: default OFF — no qwen/mai fallback (Chinese leak)
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504, 524, 529})
FALLBACK_PARTNER = {"qwen": "gemini", "gemini": "qwen"}

# --- Gemini-only + long-text chunking (Peach 2026-08-29) ---
# Peach decided 2026-08-29: drop qwen (its Thai reads are unstable/"humming" and
# it mangles mixed text), use Gemini (Kore) as THE single voice. But Gemini is
# ~20x slower than qwen on long text (a long read once took 30s → app voice
# timeout → audible error). To keep Gemini fast & stable on iOS:
#   * long text is split into chunks on sentence/space boundaries,
#   * chunks are synthesized IN PARALLEL (multiple threads against OpenRouter),
#   * the resulting pcm fragments are concatenated in order, then converted to
#     mp3 once. Parallelism caps total wall-time so a long read finishes well
#     under the app's timeout.
# GEMINI_ONLY=1 (default): EVERY /audio/speech request is served via Gemini —
# even when the app asks for qwen, the request is rewritten to Gemini. Set
# GEMINI_ONLY=0 to serve the model the app actually requests (qwen stays qwen,
# with the classic latin-strip preprocessing).
# Peach 2026-08-29 (UNIFIED single-provider architecture): every model now goes
# through THIS proxy; the app can pick any of the 7 TTS models in the one group
# and the proxy serves the model it actually requests (no cross-model rewrite),
# mapping each to a verified Thai-female voice. GEMINI_ONLY=1 (opt-in) still
# forces everything to Gemini (Kore) for the "always the same voice" mode; the
# default 0 serves the requested model as-is.
GEMINI_ONLY = os.environ.get("TTS_GEMINI_ONLY", "0") == "1"
# Peach 2026-08-31: QWEN_ONLY — every /audio/speech request is served via qwen
# flash THAI-PURE (aggressive strip + hard-locked language_type=Thai), NO other
# model is spoken. English/numbers/symbols/emoji are stripped out completely —
# only Thai is read ("ฟิกไทยล้วนโหดๆ ไม่ต้องผสม"). TTS_QWEN_ONLY=1 is default;
# set 0 to serve the model the app actually requests. Fallback (qwen errors):
# mai-voice-2 (Thai Neural) then gemini (Kore), so users never hear a raw error.
QWEN_ONLY = os.environ.get("TTS_QWEN_ONLY", "1") == "1"
MAX_CHUNK_CHARS = int(os.environ.get("TTS_MAX_CHUNK_CHARS", "180"))
MAX_WORKERS = int(os.environ.get("TTS_MAX_WORKERS", "6"))
CHUNK_TIMEOUT = int(os.environ.get("TTS_CHUNK_TIMEOUT", "60"))


def _normalize_model(model: str) -> str:
    """Strip any provider/instance prefix so we get the bare OpenRouter id.

    The app may send "<instance-id>/google/gemini-3.1-flash-tts-preview" or the
    bare "google/gemini-3.1-flash-tts-preview". instance id can be a full 36-char
    UUID **or a short 8-hex id** (e.g. Jijii POS = 9B771524). Drop a leading
    bare hex-id segment if present; keep vendor/model otherwise.
    """
    m = str(model or "").strip()
    parts = m.split("/")
    if len(parts) >= 2:
        head = parts[0].strip()
        # full UUID like 9b771524-abcd-1234-ef56-789abcdef012  OR  8-hex like 9B771524
        if re.fullmatch(r"[0-9A-Fa-f]{8}(-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}", head) or \
           re.fullmatch(r"[0-9A-Fa-f]{8}", head):
            parts = parts[1:]
        core = "/".join(parts)
        # Keep only vendor/model (drop accidental trailing :suffix like :free/:online)
        return core
    return m


def _matches_supported(model: str) -> bool:
    return any(f in model for f in SUPPORTED_FRAGMENTS)


def _strip_latin_for_qwen(text: str) -> str:
    """Thai-PURE preprocessing for qwen (Peach 2026-08-30 AGGRESSIVE).

    qwen drops all Thai when the text mixes in ANY latin word (model
    limitation, verified 2026-08-28: '...เมนู Read ...' → 1.1s Thai dropped;
    stripped → 5.7s full Thai read). Two modes:
      * AGGRESSIVE (default, 2026-08-30): keep ONLY Thai characters
        (U+0E00-U+0E7F: consonants, vowels, tone marks, Thai digits) plus
        whitespace. EVERYTHING else — latin letters, digits, punctuation,
        symbols, emoji — is removed (replaced by a single space) so qwen
        reads pure Thai.
      * CLASSIC: strip only latin word tokens.
    Pure-English text (no Thai at all) is returned unchanged: there is no Thai
    to protect, so we never gut an all-English message."""
    if not (text and QWEN_SKIP_LATIN):
        return text
    has_thai = any(0x0E00 <= ord(c) <= 0x0E7F for c in text)
    if not has_thai:
        return text                      # no Thai to protect -> send as-is
    if QWEN_AGGRESSIVE_STRIP:
        out = []
        for ch in text:
            o = ord(ch)
            if 0x0E00 <= o <= 0x0E7F:
                out.append(ch)           # Thai letter / tone mark / Thai digit
            elif ch.isspace():
                out.append(" ")          # keep whitespace (collapse later)
            elif out and out[-1] != " ":
                out.append(" ")          # non-Thai char -> single separator
        cleaned = "".join(out)
        return re.sub(r" +", " ", cleaned).strip()
    # classic: strip latin word tokens only
    if not re.search(r"[A-Za-z]", text):
        return text                      # already pure Thai -> send as-is
    return re.sub(r"[A-Za-z][A-Za-z0-9'\-\.]*", " ", text).strip()


def _gemini_payload(text: str, data: dict) -> dict:
    """Build the Thai-main TTS payload (reads mixed Thai+English without dropping).
    Returns pcm if the selected model is Gemini-native, otherwise forwards the
    model's preferred response_format. Proxy converts to mp3 for the app."""
    # response_format: Gemini-native only accepts pcm; other served models
    # (qwen-plus/fish/minimax etc) accept mp3 directly.
    fmt = "pcm" if "gemini" in GEMINI_MODEL else "mp3"
    out = {
        "model": GEMINI_MODEL,      # selectable via select-tts-model.sh
        "input": text,
        "voice": _resolve_voice(data, "gemini", GEMINI_VOICE),  # Kore (female Thai, reads mixed EN+TH)
        "response_format": fmt,
    }
    if data.get("instructions"):
        out["instructions"] = data["instructions"]
    else:
        # Full Director's Notes style (updated 2026-08-31) already includes
        # the Thai-main/Language guidance — no extra suffix needed.
        out["instructions"] = DEFAULT_GEMINI_STYLE
    if data.get("speed") is not None:
        out["instructions"] = (out["instructions"]
                               + "\nDelivery pace approx "
                               + str(float(data["speed"])) + "x of normal.")
    return out


def _pcm_header_rate(ctype: str) -> int:
    m = re.search(r"rate\s*=\s*(\d+)", ctype or "")
    return int(m.group(1)) if m else 24000


def _pcm_to_mp3(pcm: bytes, rate: int) -> bytes:
    """Convert raw signed-16-bit little-endian mono PCM to mp3 via ffmpeg.

    iSH's ffmpeg returns 0 bytes when fed via stdin (pipe:0), so we write the
    pcm to a temp file and let ffmpeg read from the file (proven to work).
    """
    import tempfile
    fd, inp = tempfile.mkstemp(suffix=".pcm")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(pcm)
        out = tempfile.mktemp(suffix=".mp3")
        ff = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "s16le", "-ar", str(rate), "-ac", "1",
             "-i", inp, "-codec:a", "libmp3lame", "-q:a", "4", "-f", "mp3", out],
            capture_output=True)
        if ff.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(f"ffmpeg pcm->mp3 failed: "
                               f"{ff.stderr.decode(errors='replace')[:300]}")
        with open(out, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(inp)
        except OSError:
            pass
        try:
            os.remove(out)
        except OSError:
            pass


def _forward(payload: bytes, method: str, path: str, content_type: str):
    url = UPSTREAM + path
    headers = {
        "Authorization": "Bearer " + API_KEY,
        "Content-Type": content_type or "application/json",
        "Accept": "*/*",
    }
    req = urllib.request.Request(url, data=payload, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()
    except Exception as e:
        return 502, {"Content-Type": "application/json"}, json.dumps(
            {"error": {"message": str(e)}}).encode()


def _post_raw(url: str, payload: bytes, timeout: int = 120):
    """POST raw bytes to url; never raises — returns (status, headers, body).

    Used by the speech path so we can inspect the upstream status and decide
    whether to fall back to the partner TTS model."""
    headers = {
        "Authorization": "Bearer " + API_KEY,
        "Content-Type": "application/json",
        "Accept": "*/*",
    }
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()
    except Exception as e:
        return 502, {"Content-Type": "application/json"}, json.dumps(
            {"error": {"message": f"upstream unreachable: {e}"}}).encode()


def _is_retryable(status: int) -> bool:
    """True if the upstream status deserves a one-shot fallback to the partner
    TTS model. 429 (rate limit), 529 (provider overloaded), 524 (edge network
    timeout — the exact error the app used to surface when gemini was slow on
    long reads), other 5xx, and 408 all qualify."""
    return status in RETRYABLE_STATUS


def _log(msg: str):
    import datetime
    sys.stderr.write("[tts-proxy %s] %s\n" % (
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg))


def _chunk_text(text: str, max_chars: int = None) -> list:
    """Split long text into sentence/space-aligned chunks for parallel synth.

    Each chunk maps to ONE Gemini request; short text stays a single chunk.
    Chunk boundary prefers a terminal punctuation or whitespace so we don't cut
    mid-word/phrase. Falls back to a hard character split for very long
    unbroken runs. Returns a non-empty list of trimmed strings."""
    max_chars = max_chars or MAX_CHUNK_CHARS
    text = (text or "").strip()
    if not text or len(text) <= max_chars:
        return [text] if text else [""]

    BOUNDARY = ".!?。！？;；,，:：\n "
    chunks = []
    cur = ""
    for ch in text:
        cur += ch
        if len(cur) >= max_chars and ch in BOUNDARY:
            chunks.append(cur.strip())
            cur = ""
    if cur.strip():
        chunks.append(cur.strip())

    # Hard-split any chunk that still exceeds max (no boundary within).
    res = []
    for c in chunks:
        while len(c) > max_chars:
            res.append(c[:max_chars].strip())
            c = c[max_chars:]
        if c.strip():
            res.append(c.strip())
    return res or [text]


def _serve_gemini_sequential(data: dict, text: str):
    """Serve the FULL text as ONE gemini request (no chunking).

    Used for short text (< chunk size) and as the still-simple fallback when the
    app requests a plain short read. Returns (status, headers, body)."""
    out = _gemini_payload(text, data)
    url = UPSTREAM + "/v1/audio/speech"
    return _post_raw(url, json.dumps(out).encode(), timeout=CHUNK_TIMEOUT)


def _serve_gemini_chunked(data: dict, text: str):
    """Synthesize long text by splitting into chunks and POSTing to gemini in
    PARALLEL, then concatenating the resulting pcm fragments in order.

    Gemini returns pcm (24000 Hz, s16le, mono) — same rate for every chunk, so
    raw byte concatenation is safe. Returns (status, headers, body) with the
    concatenated pcm for the WHOLE text. On any chunk failure it returns the
    first failing status (caller falls back to qwen once, per policy)."""
    chunks = _chunk_text(text)
    if len(chunks) <= 1:
        return _serve_gemini_sequential(data, text)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    url = UPSTREAM + "/v1/audio/speech"
    results = [None] * len(chunks)
    err_status = None
    sample_rate = 24000

    def work(i):
        chunk = chunks[i]
        out = _gemini_payload(chunk, data)
        return i, _post_raw(url, json.dumps(out).encode(), timeout=CHUNK_TIMEOUT)

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(chunks))) as ex:
        futs = [ex.submit(work, i) for i in range(len(chunks))]
        for fut in as_completed(futs):
            i, (st, hd, bd) = fut.result()
            if st != 200:
                if err_status is None:
                    err_status = st
                continue
            if sample_rate == 24000:
                rate = _pcm_header_rate(hd.get("Content-Type", ""))
                if rate:
                    sample_rate = rate
            results[i] = bd

    if err_status is not None or any(r is None for r in results):
        # A chunk failed — signal error so caller can fall back.
        ctype = "application/json"
        body = json.dumps({"error": {"message": f"gemini chunk failed ({err_status})"}}).encode()
        return err_status or 502, {"Content-Type": ctype}, body

    pcm = b"".join(results)
    return 200, {"Content-Type": f"audio/pcm; rate={sample_rate}"}, pcm


def _serve_gemini(data: dict, text: str):
    """Gemini-first serving of /audio/speech with long-text parallel chunking.

    Peach 2026-08-29: Gemini (Kore) is now THE single voice (qwen dropped). This
    is the primary speech path: short text = single request; long text = split +
    parallel + concat (stays under the app timeout). Returns
    (status, headers, body, used_kind="gemini")."""
    if TTS_GEMINI_DIRECT:
        return _serve_gemini_direct_google(data, text)
    if len((text or "").strip()) <= (MAX_CHUNK_CHARS or 180):
        return _serve_gemini_sequential(data, text) + ("gemini",)
    return _serve_gemini_chunked(data, text) + ("gemini",)


# Retry-on-empty/5xx (Peach 2026-08-31): the app fires many TTS requests in a
# burst (e.g. testing every voice group); OpenRouter/Gemini intermittently
# answers 200 with an EMPTY audio body -> proxy correctly reports
# 502 "Provider returned an empty audio stream". With FALLBACK=0 (single-voice
# mode) that 502 went straight to the app -> "Voice synthesis failed" banner
# even though other requests in the same burst played fine. Fix: retry the
# SAME gemini request a couple of times (no cross-model fallback), which
# absorbs the transient empty-stream / 429 / 5xx.
TTS_RETRIES = int(os.environ.get("TTS_RETRIES", "2"))      # extra tries after 1st
TTS_RETRY_DELAY = float(os.environ.get("TTS_RETRY_DELAY", "0.4"))


def _serve_gemini_retry(data: dict, text: str):
    """Try _serve_gemini up to 1+TTS_RETRIES times (retry only on retryable
    status codes; 200/400/401/402... are returned as-is)."""
    status, headers, body, used = _serve_gemini(data, text)
    for attempt in range(TTS_RETRIES):
        if status == 200 or not _is_retryable(status):
            break
        time.sleep(TTS_RETRY_DELAY * (attempt + 1))
        _log(f"gemini retry {attempt+1}/{TTS_RETRIES} (status {status})")
        status, headers, body, used = _serve_gemini(data, text)
    if status == 200:
        return status, headers, body, used
    _log(f"gemini gave up after {TTS_RETRIES} retries (status {status})")
    return status, headers, body, used


def _gemini_direct_model(model: str) -> str:
    """OpenRouter-style 'google/gemini-3.1-flash-tts-preview' -> Gemini model id."""
    m = model.split("/")[-1]
    return m


def _serve_gemini_direct_google(data: dict, text: str):
    """Serve gemini TTS POSTing generateContent directly to Google (no
    OpenRouter). Uses $GEMINI_API_KEY + GEMINI_MODEL/GEMINI_VOICE. Returns
    (status, headers, pcm_body, "gemini") — caller converts pcm->mp3."""
    if not GEMINI_API_KEY:
        return 502, {"Content-Type": "application/json"}, json.dumps(
            {"error": {"message": "TTS_GEMINI_DIRECT=1 but GEMINI_API_KEY not set"}}).encode(), "gemini"
    model = _gemini_direct_model(GEMINI_MODEL)
    voice = GEMINI_VOICE
    inst = data.get("instructions") or DEFAULT_GEMINI_STYLE
    payload = {
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "systemInstruction": {"parts": [{"text": inst}]},
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }
    url = "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":generateContent?key=" + GEMINI_API_KEY
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Accept": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=CHUNK_TIMEOUT) as resp:
            j = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        _log(f"gemini-direct upstream {e.code}: {body[:300]}")
        return e.code, dict(e.headers), body.encode(), "gemini"
    except Exception as e:
        _log(f"gemini-direct unreachable: {e}")
        return 502, {"Content-Type": "application/json"}, json.dumps(
            {"error": {"message": f"gemini-direct unreachable: {e}"}}).encode(), "gemini"
    audio = b""
    try:
        for cand in j.get("candidates", []):
            for part in (cand.get("content", {}).get("parts", []) or []):
                if part.get("inlineData", {}).get("data"):
                    audio = base64.b64decode(part["inlineData"]["data"])
                    break
    except Exception as e:
        _log(f"gemini-direct parse err: {e}")
    if not audio:
        m2 = json.dumps({"error": {"message": "gemini-direct: no audio in response"}, "resp": j})
        return 502, {"Content-Type": "application/json"}, m2.encode(), "gemini"
    return 200, {"Content-Type": "audio/pcm"}, audio, "gemini"


def _fish_direct_synthesize(text: str, reference_id: str, model_hint: str = "") -> bytes:
    """Serve fish TTS POSTing /v1/tts DIRECTLY to fish.audio (no OpenRouter —
    OpenRouter dropped fish TTS, verified 2026-09-04). Uses $FISH_API_KEY +
    $FISH_API_MODEL (default s2.1-pro-free = free tier, no API credit).
    Body: {text, reference_id, format: mp3}. Returns mp3 bytes.

    Stability (2026-09-05 scan): retries on transient fish errors (429/5xx,
    per official docs "wait and retry with backoff") using the same
    RETRYABLE_STATUS + TTS_RETRIES as Gemini; separate FISH_TIMEOUT for long
    text; explicit 402 credit hint; quota counted only after success."""
    if not FISH_API_KEY:
        raise RuntimeError("FISH_API_KEY / AZURE_API_KEY not set (fish key needed)")
    if not re.fullmatch(r"[0-9a-f]{32}", reference_id):
        reference_id = FISH_FREE_VOICE
    # 2026-09-05 voice-design fix (Peach: "เสียงเพี้ยน/พูดไม่ชัด"): raw app text
    # (numbers like 1,500 / ฿ / 0xx phones / mixed EN) reads distorted -> run
    # our Thai normalizer first, then pin the official stability params
    # (temperature low, repetition_penalty >=1.2, prosody.speed <1.0,
    # normalize, condition_on_previous_chunks) per docs.fish.audio endpoint spec.
    orig_len = len(text)
    text = _thai_normalize_text(text) if FISH_NORMALIZE_TH else text
    payload = {
        "text": text,
        "reference_id": reference_id,
        "format": "mp3",
        "latency": FISH_LATENCY,
        "temperature": FISH_TEMPERATURE,
        "top_p": FISH_TOP_P,
        "repetition_penalty": FISH_REPETITION_PENALTY,
        "normalize": FISH_NORMALIZE,
        "chunk_length": min(300, max(100, FISH_CHUNK_LENGTH)),
        "condition_on_previous_chunks": True,
        "prosody": {"speed": FISH_SPEED, "normalize_loudness": True},
    }
    if FISH_QUALITY_GUARD:
        payload["features"] = ["quality-guard"]
    body = None
    last_status = None
    last_msg = ""
    for attempt in range(TTS_RETRIES + 1):
        try:
            req = urllib.request.Request(
                "https://api.fish.audio/v1/tts",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": "Bearer " + FISH_API_KEY,
                    "Content-Type": "application/json",
                    "model": FISH_API_MODEL,
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=FISH_TIMEOUT) as resp:
                b = resp.read()
            if not b or len(b) < 1000:
                raise RuntimeError("fish direct: unexpected short body (%d bytes)" % len(b))
            body = b
            break
        except urllib.error.HTTPError as e:
            last_status = e.code
            last_msg = e.read().decode("utf-8", "replace")[:200]
            if not _is_retryable(e.code):
                break
        except Exception as e:
            last_msg = str(e)[:200]
        if attempt < TTS_RETRIES:
            time.sleep(TTS_RETRY_DELAY * (attempt + 1))
    if body is None:
        if last_status == 402:
            raise RuntimeError("fish out of credits (402) — top up at fish.audio/app/billing")
        raise RuntimeError("fish direct failed (http=%s): %s" % (last_status, last_msg))
    _fish_count_chars(orig_len)      # count original input length after success
    return body


def _fish_ws_stream_chunks(text: str, reference_id: str, model: str):
    """Stream fish TTS via WebSocket /v1/tts/live, yielding mp3 chunks as they
    arrive (real-time). Uses websockets.sync + msgpack directly (fish SDK won't
    build on iSH musl). Peach 2026-09-04: verified working with $FISH_API_KEY,
    s2.1-pro-free, เสียงพี่ reference 6eb7....
    Wire: StartEvent->TextEvent...->FlushEvent->StopEvent
          <- AudioEvent(bin mp3) xN -> FinishEvent(reason).
    """
    import websockets.sync.client as wsync
    import msgpack
    if not FISH_API_KEY:
        raise RuntimeError("FISH_API_KEY not set (fish key needed)")
    if not re.fullmatch(r"[0-9a-f]{32}", reference_id):
        reference_id = FISH_FREE_VOICE
    uri = "wss://api.fish.audio/v1/tts/live"
    with wsync.connect(
        uri,
        additional_headers={"Authorization": "Bearer " + FISH_API_KEY, "model": model},
        open_timeout=25, close_timeout=8, max_size=20 * 1024 * 1024,
    ) as ws:
        start = {"event": "start", "request": {
            "text": "", "format": "mp3", "chunk_length": min(300, max(100, FISH_CHUNK_LENGTH)),
            "reference_id": reference_id, "latency": FISH_LATENCY,
            # 2026-09-05 voice-design fix: same stability knobs as the REST path
            "temperature": FISH_TEMPERATURE, "top_p": FISH_TOP_P,
            "repetition_penalty": FISH_REPETITION_PENALTY,
            "normalize": FISH_NORMALIZE,
            "condition_on_previous_chunks": True,
            "prosody": {"speed": FISH_SPEED, "normalize_loudness": True},
        }}
        ws.send(msgpack.packb(start, use_bin_type=True))
        # split text at breath boundaries (no mid-word cuts — the old fixed 60-char
        # step split Thai words and caused slurred pronunciation), then send text
        # and trigger flush per chunk for real-time; flush+stop at end.
        text = _thai_normalize_text(text) if FISH_NORMALIZE_TH else text
        parts = _thai_split_sentences(text, min(300, max(100, FISH_CHUNK_LENGTH)), 20)
        for idx, p in enumerate(parts):
            ws.send(msgpack.packb({"event": "text", "text": p}, use_bin_type=True))
            if idx < len(parts) - 1:
                ws.send(msgpack.packb({"event": "flush"}, use_bin_type=True))
        ws.send(msgpack.packb({"event": "flush"}, use_bin_type=True))
        ws.send(msgpack.packb({"event": "stop"}, use_bin_type=True))
        # collect audio events
        n = 0
        for raw in ws:
            obj = msgpack.unpackb(raw)
            ev = obj.get("event") if isinstance(obj, dict) else None
            if ev == "audio":
                chunk = obj["audio"]
                if isinstance(chunk, str):
                    chunk = chunk.encode("latin1")
                n += 1
                yield chunk
            elif ev == "finish":
                reason = obj.get("reason")
                if reason != "stop":
                    sys.stderr.write("[tts-proxy] fish ws finish=%s after %d frames\n" % (reason, n))
                else:
                    # count quota only when the stream finished cleanly
                    _fish_count_chars(len(text))
                return


def _serve_speech_with_fallback(payload: bytes, kind: str, data: dict):
    """Serve ONE /audio/speech request — GEMINI-FIRST with error-aware fallback.

    Peach 2026-08-29: Gemini (Kore) is THE single voice. With GEMINI_ONLY=1
    (default) every request is served via Gemini with long-text parallel
    chunking (fast, under the app timeout) — even qwen requests are rewritten
    to Gemini. Only if Gemini fails with a retryable error do we fall back ONCE
    to qwen as a last resort so the user never sees a raw error.
    With GEMINI_ONLY=0 the model the app requested is served as-is.

    Returns (status, headers, body, used_kind). The caller converts pcm->mp3
    when used_kind == "gemini"."""
    text = data.get("input", "")
    url = UPSTREAM + "/v1/audio/speech"

    # UNIFIED single-provider (Peach 2026-08-29): serve the model the app
    # requested, its own kind, with its verified Thai-female voice. No
    # cross-model rewrite (except a smart one-shot error fallback to a Thai
    # voice partner so Peach never hears a raw error). GEMINI_ONLY=1 forces all
    # traffic to Gemini (Kore) for the "always the same voice" mode.

    # QWEN_ONLY=1 (Peach 2026-08-31 "ฟิกไทยล้วนโหดๆ"): EVERY request is served
    # via qwen flash Thai-PURE — aggressive strip removes all non-Thai (english/
    # numbers/symbols/emoji), hard-locked language_type=Thai, only Thai is read.
    # Qwen ล้ม = คืน error ตรง ๆ (NO model fallback — Peach 2026-08-31):
    # กลุ่ม TTS ทำ failover ระหว่าง 2 พอร์ต (8765 -> 8766) ที่เป็น qwen ทั้งคู่
    # แทนที่จะได้เสียง mai/gemini แทรกมา
    if QWEN_ONLY:
        try:
            pb = json.dumps(_qwen_payload(_strip_latin_for_qwen(text), data)).encode()
            _st, _hd, _bd = _post_raw(url, pb, timeout=CHUNK_TIMEOUT)
            if _st != 200:
                # Peach 2026-08-31: dump the ACTUAL outbound payload on failure
                # so we can see which field/input makes OpenRouter answer 400.
                _log(f"qwen-only upstream {_st} | orig-input={str(data.get('input'))[:200]!r} "
                     f"| keys={sorted(str(k) for k in data.keys())} "
                     f"| PAYLOAD={pb.decode('utf-8','replace')[:600]}")
            return _st, _hd, _bd, "qwen"
        except Exception as e:
            _log(f"qwen-only build failed: {e}")
            return 502, {"Content-Type": "application/json"}, json.dumps(
                {"error": {"message": f"qwen-only build failed: {e}"}}).encode(), "qwen"

    # GEMINI_ONLY=1 (opt-in single-voice mode): everything via Gemini.
    if GEMINI_ONLY:
        if TTS_GEMINI_DIRECT:
            status, headers, body, used = _serve_gemini_direct_google(data, text)
        else:
            status, headers, body, used = _serve_gemini_retry(data, text)
        if status == 200:
            return status, headers, body, used
        # Peach 2026-08-31: dump EXACT outbound payload on failure so we can
        # see which app field/probe makes OpenRouter answer 400 (app sends
        # inlen 29/59 -> 400, same text direct -> 200).
        _log(f"gemini-only upstream {status} | orig-input={str(data.get('input'))[:200]!r} "
             f"| keys={sorted(str(k) for k in data.keys())} "
             f"| PAYLOAD={json.dumps(_gemini_payload(text, data)).encode('utf-8','replace')[:600]}")
        if FALLBACK_ENABLED and _is_retryable(status):
            # gemini failed -> mai-voice-2 (Thai Neural, reads mixed) first.
            try:
                pb = json.dumps(_lab_payload(MAI_MODEL, _strip_latin_for_qwen(text), data, "azure")).encode()
            except Exception as e:
                _log(f"fallback mai build failed: {e}")
                pb = None
            if pb:
                s2, h2, b2 = _post_raw(url, pb, timeout=CHUNK_TIMEOUT)
                if s2 == 200:
                    _log("fallback OK via mai-voice-2 (Premwadee, Thai Neural)")
                    return s2, h2, b2, "azure"
            try:
                pb = json.dumps(_qwen_payload(_strip_latin_for_qwen(text), data)).encode()
            except Exception as e:
                _log(f"fallback qwen build failed: {e}")
                return status, headers, body, used
            _log(f"gemini-only fallback gemini->mai/qwen (primary status {status})")
            s2, h2, b2 = _post_raw(url, pb, timeout=CHUNK_TIMEOUT)
            if s2 == 200:
                _log("fallback OK via qwen (Thai female)")
                return s2, h2, b2, "qwen"
            _log(f"fallback qwen also failed ({s2}) — returning primary error")
        return status, headers, body, used

    # Serve the requested model directly.
    if kind == "gemini":
        status, headers, body, used = _serve_gemini_retry(data, text)
        if status == 200:
            return status, headers, body, used
        # Peach 2026-08-31: dump EXACT outbound payload on failure so we can
        # see which app field/input makes OpenRouter answer 400.
        _log(f"gemini upstream {status} | orig-input={str(data.get('input'))[:200]!r} "
             f"| keys={sorted(str(k) for k in data.keys())} "
             f"| PAYLOAD={json.dumps(_gemini_payload(text, data)).encode('utf-8','replace')[:600]}")
        if FALLBACK_ENABLED and _is_retryable(status):
            # gemini failed -> mai-voice-2 (Thai Neural, reads mixed) first.
            try:
                pb = json.dumps(_lab_payload(MAI_MODEL, _strip_latin_for_qwen(text), data, "azure")).encode()
            except Exception as e:
                _log(f"fallback mai build failed: {e}")
                pb = None
            if pb:
                s2, h2, b2 = _post_raw(url, pb, timeout=CHUNK_TIMEOUT)
                if s2 == 200:
                    _log("fallback OK via mai-voice-2 (Premwadee, Thai Neural)")
                    return s2, h2, b2, "azure"
            try:
                pb = json.dumps(_qwen_payload(_strip_latin_for_qwen(text), data)).encode()
            except Exception as e:
                _log(f"fallback qwen build failed: {e}")
                return status, headers, body, used
            _log(f"gemini->mai/qwen fallback (primary status {status})")
            s2, h2, b2 = _post_raw(url, pb, timeout=CHUNK_TIMEOUT)
            if s2 == 200:
                _log("fallback OK via qwen (Thai female)")
                return s2, h2, b2, "qwen"
            _log(f"fallback qwen also failed ({s2}) — returning primary error")
        return status, headers, body, used

    # qwen / fish / minimax / azure — serve as requested (payload already built
    # with the correct Thai-female voice per model). Retry the SAME model on
    # transient empty-stream / 5xx (no cross-model fallback — Peach 2026-08-31).
    status, headers, body = _post_raw(url, payload, timeout=CHUNK_TIMEOUT)
    for attempt in range(TTS_RETRIES):
        if status == 200 or not _is_retryable(status):
            break
        time.sleep(TTS_RETRY_DELAY * (attempt + 1))
        _log(f"{kind} retry {attempt+1}/{TTS_RETRIES} (status {status})")
        status, headers, body = _post_raw(url, payload, timeout=CHUNK_TIMEOUT)
    if status == 200:
        return status, headers, body, kind
    if FALLBACK_ENABLED and _is_retryable(status):
        _log(f"{kind}->fallback (primary status {status})")
        if kind == "qwen":
            # qwen failed -> mai-voice-2 first (Thai Neural female, reads Thai
            # clearly, less Chinese leakage than qwen), then gemini (Kore) last.
            try:
                pb = json.dumps(_lab_payload(MAI_MODEL, _strip_latin_for_qwen(text), data, "azure")).encode()
            except Exception as e:
                _log(f"fallback mai build failed: {e}")
                pb = None
            if pb:
                s2, h2, b2 = _post_raw(url, pb, timeout=CHUNK_TIMEOUT)
                if s2 == 200:
                    _log("fallback OK via mai-voice-2 (Premwadee, Thai Neural)")
                    return s2, h2, b2, "azure"
                _log(f"fallback mai failed ({s2}) — trying gemini")
            try:
                s2, h2, b2, fu = _serve_gemini(data, text)
            except Exception as e:
                _log(f"fallback gemini build failed: {e}")
                return status, headers, body, kind
            if s2 == 200:
                _log("fallback OK via gemini (Kore, female)")
                return s2, h2, b2, "gemini"
            _log("fallback qwen->mai/gemini all failed")
        else:
            # fish / minimax / azure failed -> qwen (Thai-lock female).
            try:
                pb = json.dumps(_qwen_payload(_strip_latin_for_qwen(text), data)).encode()
            except Exception as e:
                _log(f"fallback qwen build failed: {e}")
                return status, headers, body, kind
            s2, h2, b2 = _post_raw(url, pb, timeout=CHUNK_TIMEOUT)
            if s2 == 200:
                _log("fallback OK via qwen (Thai female)")
                return s2, h2, b2, "qwen"
        _log(f"fallback for {kind} failed")
    return status, headers, body, kind


def _qwen_payload(text: str, data: dict) -> dict:
    """Build the qwen TTS payload (THE main Thai voice since 2026-08-28).

    Peach 2026-08-30 (AGGRESSIVE Thai-only): model from QWEN_MODEL env
    (default qwen/qwen-audio-3.0-tts-flash), voice longanhuan_v3.6, and
    language_type hard-locked to "Thai" ALWAYS so qwen never drifts to
    English/Chinese.

    Peach 2026-08-31 (root cause của "Voice synthesis failed"): qwen does NOT
    support the `speed` parameter — OpenRouter returns 400 "Alibaba Qwen TTS
    does not support the speed parameter" (verified: speed=0.9 → 400). The app
    sends a speed value ≠ 1, which killed every read. FIX: NEVER pass speed to
    qwen (omit it entirely; qwen then uses default pace). instructions is fine
    (200) and kept."""
    out = {
        "model": QWEN_MODEL,
        "input": text,
        "voice": _resolve_voice(data, "qwen", QWEN_VOICE),  # longanhuan_v3.6 (female, clear Thai)
        "response_format": "mp3",
        "language_type": "Thai",   # hard-locked (Peach 2026-08-30)
    }
    # NOTE: `speed` intentionally NOT passed — qwen rejects it (400). Omit.
    if data.get("instructions"):
        out["instructions"] = data["instructions"]
    return out


def _resolve_voice(data: dict, kind: str, default_voice: str) -> str:
    """Pick a provider-safe voice for the request.

    The app sends voice="default" (or "auto"/"none" etc.) when the user wants
    the model's default voice. fish / azure reject unknown voice strings with
    HTTP 400 ("Provider returned 400"), which kills the whole read. Rule: a
    request voice is honoured ONLY if it looks like a real voice token of that
    provider; placeholders (default/auto/none/empty) and unrecognised tokens
    fall back to the pinned Thai-female env voice (verified 2026-08-29)."""
    v = str(data.get("voice") or "").strip()
    if not v or v.lower() in ("default", "auto", "none", "system", "model"):
        return default_voice
    vl = v.lower()
    if kind == "fish":
        # fish reference_ids are 32-char lowercase hex (e.g. Cyn Thai a640c0...).
        return v if re.fullmatch(r"[0-9a-f]{32}", v) else default_voice
    if kind == "azure":
        # azure-style: 'th-TH-Premwadee:MAI-Voice-2' / 'Microsoft Server Speech ...'
        return v if (v.startswith("th-TH-")
                     or "mai-voice" in vl
                     or v.startswith("Microsoft Server Speech")) else default_voice
    if kind == "minimax":
        # official Thai presets: 'Thai_...' / 'MiniMax/.../Thai_' / 'female-thai...'
        return v if ("thai" in vl) else default_voice
    if kind == "qwen":
        return v  # qwen accepts arbitrary names; placeholder already handled
    if kind == "gemini":
        # Gemini named voices ONLY (Peach 2026-08-31; expanded to full 30-voice
        # catalog 2026-08-31 after live E2E test through OpenRouter — all 30
        # returned 200). The app may pass a STALE voice string from another
        # provider (qwen longanhuan_v3.6 / azure th-TH-Premwadee / minimax
        # Thai_tender_woman...), and OpenRouter Gemini TTS rejects those with
        # 400 "Provider returned 400" -> the app shows "Voice synthesis failed".
        # Whitelist the 30 official Gemini 3.1 TTS voices (Google docs):
        # Zephyr(Bright) Puck(Upbeat) Charon(Informative) Kore(Firm) Fenrir
        # (Excitable) Leda(Youthful) Orus(Firm) Aoede(Breezy) Callirrhoe
        # (Easy-going) Autonoe(Bright) Enceladus(Breathy) Iapetus(Clear)
        # Umbriel(Easy-going) Algieba(Smooth) Despina(Smooth) Erinome(Clear)
        # Algenib(Gravelly) Rasalgethi(Informative) Laomedeia(Upbeat)
        # Achernar(Soft) Alnilam(Firm) Schedar(Even) Gacrux(Mature)
        # Pulcherrima(Forward) Achird(Friendly) Zubenelgenubi(Casual)
        # Vindemiatrix(Gentle) Sadachbia(Lively) Sadaltager(Knowledgeable)
        # Sulafat(Warm). Anything else pins the Thai-female GEMINI_VOICE.
        GEMINI_VOICES = {
            "zephyr","puck","charon","kore","fenrir","leda","orus","aoede",
            "callirrhoe","autonoe","enceladus","iapetus","umbriel","algieba",
            "despina","erinome","algenib","rasalgethi","laomedeia","achernar",
            "alnilam","schedar","gacrux","pulcherrima","achird","zubenelgenubi",
            "vindemiatrix","sadachbia","sadaltager","sulafat",
        }
        return v if vl in GEMINI_VOICES else default_voice
    return v     # gemini named voices (Kore, Achernar, ...); placeholder handled


def _lab_payload(model: str, text: str, data: dict, kind: str) -> dict:
    """Build the correct payload for a Voice Lab model (non-gemini/non-qwen).

    Each provider has its own voice/speed contract. voice defaults are chosen
    from the OpenRouter TTS guide; qwen-plus reuses the qwen Thai-lock path."""
    out = {
        "model": model,
        "input": text,
        "response_format": "mp3",
    }
    if kind == "fish":
        # fish: Thai-FEMALE voice pinned via env (verified 29/8). "Jijii voice"
        # (80e9375d...) is the Fish.audio voice-model Jijii (created 5/9) used
        # for BOTH tiers: s2.1-pro (pro) and s2.1-pro-free:free (free) accept it
        # via fish direct API (E2E 200, distinct audio). Uses FISH_VOICE for
        # s2.1-pro and FISH_FREE_VOICE for the free tier — both point to the
        # same Jijii voice id so the voice NEVER flips between tiers. Request
        # voice honoured only if it is a real 32-hex fish reference_id (else
        # fish returns 400 "Provider returned 400").
        if "free" in model:
            out["voice"] = _resolve_voice(data, "fish", FISH_FREE_VOICE)
        else:
            out["voice"] = _resolve_voice(data, "fish", FISH_VOICE)
    elif kind == "minimax":
        # Thai-female preset via MINIMAX_VOICE env (verified 29/8). Official
        # MiniMax Thai female voices: Thai_confident_woman / Thai_optimistic_girl /
        # Thai_energetic_woman / Thai_tender_woman.
        out["voice"] = _resolve_voice(data, "minimax", MINIMAX_VOICE)
    elif kind == "azure":
        # mai-voice-2 / Azure-style: Thai-female Neural (Premwadee) via MAI_VOICE env.
        out["voice"] = _resolve_voice(data, "azure", MAI_VOICE)
    if data.get("speed") is not None:
        out["speed"] = float(data["speed"])
    return out


def _build_speech_payload(data: dict):
    """
    Rewrite /audio/speech for OpenRouter. Correct TTS call pattern (matches the
    OpenRouter Python/TS SDK docs + 2026-08-27 guide):
      - qwen   : {model, input, voice=longanhuan_v3.6, response_format=mp3} (+Thai-lock)
      - gemini : {model, input, voice=Kore, response_format=pcm}  (pcm only!)
      - lab    : fish/minimax/azure -> per-provider payload from _lab_payload,
                 served on the Voice Lab port (:8766) without Gemini forcing.

    Peach 2026-08-29 layout = Gemini-ONLY single voice (Kore) for PRODUCTION.
    GEMINI_ONLY=1 (default) means speech on the production port is served via
    Gemini regardless of the model the app requests. The Voice Lab port
    (:8766, TTS_PROXY_PORT) disables Gemini-forcing (GEMINI_ONLY treated as 0)
    so the app can actually A/B the other Thai TTS models.
    """
    model = _normalize_model(data.get("model", ""))
    text = data.get("input", "")

    # 4/9: qwen REMOVED (Chinese model → เสียงจีนปน) — fish/gemini/gcp only.

    if "gemini" in model and "tts" in model:
        # covers gemini-2.5-flash-preview-tts AND gemini-3.1-flash-tts-preview
        # (Peach 3/9: two choices, direct to Google, no OpenRouter).
        out = _gemini_payload(text, data)
        return out, "gemini"

    # Voice Lab / direct models — per-provider payload.
    if "fish-audio/s2.1-pro" in model:
        # 4/9: fish goes DIRECT to api.fish.audio (OR dropped fish TTS).
        return {"model": model, "input": text}, "fish_direct"

    # Google Cloud TTS — th-TH-Neural2-C/A (female Thai, Peach 3/9 policy: the
    # ONLY TTS that actually works for Thai). Synthesized DIRECT at
    # texttospeech.googleapis.com via FIREBASE_SERVICE_ACCOUNT (not OpenRouter —
    # OpenRouter has no Google TTS; qwen/mai/minimax didn't work in the field).
    if "th-TH-Neural2" in model:
        return {"model": model, "input": text}, "gcp"

    # Not a served TTS model — fall through as literal passthrough of original.
    return dict(data), None


def _gcp_synthesize(text: str, voice: str = "th-TH-Neural2-C", rate: float = 1.0) -> bytes:
    """Google Cloud TTS direct via service account (FIREBASE_SERVICE_ACCOUNT env).
    Returns mp3 bytes. Female-Thai voices only: th-TH-Neural2-C / th-TH-Neural2-A.
    (Same path as memory/tools/tts/cloud_tts_direct.py — used by the app speaker.)
    Quota guard (REBUILD-2): blocks once used+len(text) >= CLOUD_MONTHLY_LIMIT."""
    ok, used, limit, pct = _gcp_quota_check(len(text or ""))
    if not ok:
        raise RuntimeError(
            f"Google Cloud TTS monthly quota reached ({used}/{limit} chars) — "
            "raise CLOUD_MONTHLY_LIMIT or switch to the fish voice group")
    import base64 as _b64
    from google.oauth2 import service_account as _sa
    from google.auth.transport.requests import Request as _GReq
    sa_raw = os.environ.get("FIREBASE_SERVICE_ACCOUNT", "")
    if not sa_raw:
        raise RuntimeError("FIREBASE_SERVICE_ACCOUNT not set")
    sa = json.loads(sa_raw)
    creds = _sa.Credentials.from_service_account_info(
        sa, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(_GReq())
    payload = {
        "input": {"text": text},
        "voice": {"languageCode": voice.split("-")[0] + "-" + voice.split("-")[1],
                  "name": voice},
        "audioConfig": {"audioEncoding": "MP3", "speakingRate": rate, "pitch": 0.0},
    }
    req = urllib.request.Request(
        "https://texttospeech.googleapis.com/v1/text:synthesize",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + creds.token,
                 "Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=60)
    audio = _b64.b64decode(json.loads(resp.read().decode("utf-8"))["audioContent"])
    _gcp_count_chars(len(text or ""))
    return audio


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self): self._handle()
    def do_POST(self): self._handle()
    def do_HEAD(self): self._handle()

    def log_message(self, fmt, *args):
        # Structured, timestamped log so we can correlate a fetch with the
        # exact time/endpoint — raw HTTP tracebacks alone can't map to "when the
        # user pressed read-aloud".
        import datetime
        try:
            self.server.log_time = datetime.datetime.now().strftime("%H:%M:%S")
            sys.stderr.write("[tts-proxy %s] %s %s\n" % (
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.command, self.path))
        except Exception:
            pass

    def _handle(self):
        try:
            import time as _time_mod
            _t0 = _time_mod.time()
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            path = self.path
            upstream_path = path if path.startswith("/") else "/" + path
            if not upstream_path.startswith("/v1"):
                upstream_path = "/v1" + upstream_path

            is_live = upstream_path.endswith("/tts/live") and self.command == "POST"
            is_speech = upstream_path.endswith("/audio/speech") and self.command == "POST"
            payload = raw
            convert_pcm = False
            speech_supported = False
            used_kind = None

            # ANY port: answer /models with our 7-model TTS catalog so the app
            # sees ALL voice models through one proxy/provider (not just the 5
            # that OpenRouter's plain /models used to surface). OpenRouter's
            # plain /models doesn't list TTS models (only ?output_modalities=
            # speech does), which is why app models showed "no longer
            # available". Serving LAB_CATALOG on every port fixes that for the
            # unified single-provider layout.
            if (self.command == "GET"
                    and (path == "/models" or upstream_path.endswith("/models"))):
                bm = json.dumps({"data": LAB_CATALOG, "object": "list"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(bm)))
                self.end_headers()
                self.wfile.write(bm)
                return

            # 4/9: POST /v1/tts/live — fish WebSocket REAL-TIME streaming.
            # Proxy opens WS -> fish live, sends text chunks, streams mp3 chunks
            # back with Transfer-Encoding: chunked (app plays audio as it arrives,
            # before generation finishes). Payload: {input, voice?, model?}.
            # Voice = Peach's reference id (6eb7...) if valid, else default.
            if is_live and raw:
                try:
                    data = json.loads(raw or b"{}")
                    text = str(data.get("input", "") or data.get("text", ""))
                    vid = str(data.get("voice") or "")
                    model = _normalize_model(data.get("model", ""))
                    # voice by tier (pro -> FISH_VOICE, free -> FISH_FREE_VOICE)
                    default_vid = FISH_FREE_VOICE if "free" in model else FISH_VOICE
                    if not re.fullmatch(r"[0-9a-f]{32}", vid):
                        vid = default_vid
                    if "free" in model.lower():
                        model = FISH_API_MODEL   # s2.1-pro-free
                    elif model and ("s2" in model.lower() or "s1" in model.lower()):
                        model = model          # keep s2-pro etc
                    else:
                        model = FISH_API_MODEL
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/mpeg")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.send_header("X-Fish-Stream", "live")
                    self.end_headers()
                    sent = 0
                    try:
                        for chunk in _fish_ws_stream_chunks(text, vid, model):
                            if not chunk:
                                continue
                            sz = len(chunk)
                            self.wfile.write(("%x\r\n" % sz).encode("ascii"))
                            self.wfile.write(chunk)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                            sent += sz
                    finally:
                        try:
                            self.wfile.write(b"0\r\n\r\n")
                            self.wfile.flush()
                        except Exception:
                            pass
                    sys.stderr.write(
                        "[tts-proxy %s] /tts/live streamed OK model=%s %dB\n"
                        % (time.strftime("%Y-%m-%d %H:%M:%S"), model, sent))
                    return
                except Exception as e:
                    body = json.dumps({"error": {"message": "fish ws live failed: %s" % e}}).encode()
                    try:
                        self.send_response(502)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                    except Exception:
                        pass
                    return

            if is_speech and raw:
                try:
                    data = json.loads(raw or b"{}")
                except Exception:
                    data = None
                if data:
                    try:
                        out, kind = _build_speech_payload(data)
                        model = _normalize_model(data.get("model", ""))
                        if _matches_supported(model):
                            speech_supported = True
                            payload = json.dumps(out).encode()
                    except Exception as e:
                        payload = json.dumps(
                            {"error": {"message": f"payload build failed: {e}"}}).encode()

            if speech_supported and data:
                if kind == "fish_direct":
                    # 4/9: fish TTS ส่งตรง api.fish.audio (ไม่ผ่าน OR — OR ถอน
                    # fish ออกแล้ว). reference_id = เสียงจีจี้ default.
                    # 5/9 stable: pick voice by TIER — pro -> FISH_VOICE,
                    # free -> FISH_FREE_VOICE (both = Jijii 80e9375d now, but
                    # keeps them independent if ever set differently).
                    try:
                        text = str(data.get("input", ""))
                        vid = str(data.get("voice") or "")
                        default_vid = FISH_FREE_VOICE if "free" in model else FISH_VOICE
                        if not re.fullmatch(r"[0-9a-f]{32}", vid):
                            vid = default_vid
                        out = _fish_direct_synthesize(text, vid, model)
                        status, headers = 200, {"Content-Type": "audio/mpeg"}
                        used_kind = "fish_direct"
                    except urllib.error.HTTPError as e:
                        body = e.read().decode("utf-8", "replace")
                        out = json.dumps({"error": {"message": f"fish direct {e.code}: {body[:300]}"}}).encode()
                        status, headers = e.code, {"Content-Type": "application/json"}
                        used_kind = "fish_direct"
                    except Exception as e:
                        out = json.dumps({"error": {"message": f"fish direct failed: {e}"}}).encode()
                        status, headers = 502, {"Content-Type": "application/json"}
                        used_kind = "fish_direct"
                elif kind == "gcp":
                    # Google Cloud TTS direct (th-TH-Neural2-C/A) — Peach 3/9:
                    # the ONLY TTS that works for Thai. Skip OpenRouter entirely.
                    try:
                        want = str(data.get("voice") or "")
                        m = _normalize_model(data.get("model", ""))
                        voice = (want if want.startswith("th-TH-Neural2")
                                 else (m if m.startswith("th-TH-Neural2") else "th-TH-Neural2-C"))
                        if voice not in ("th-TH-Neural2-C", "th-TH-Neural2-A"):
                            voice = "th-TH-Neural2-C"
                        if voice == "th-TH-Neural2-A":
                            # th-TH-Neural2-A doesn't exist on Google (Thai Neural2
                            # has only C female / D male) — pin to C to avoid 502.
                            voice = "th-TH-Neural2-C"
                        out = _gcp_synthesize(str(data.get("input", "")), voice)
                        status, headers = 200, {"Content-Type": "audio/mpeg"}
                        used_kind = "gcp"
                    except Exception as e:
                        out = json.dumps({"error": {"message": f"gcp synth failed: {e}"}}).encode()
                        status, headers = 502, {"Content-Type": "application/json"}
                        used_kind = "gcp"
                else:
                    status, headers, out, used_kind = _serve_speech_with_fallback(
                        payload, kind, data)
                # gemini returns pcm — convert to mp3 for the app (also covers
                # a qwen->gemini fallback, where used_kind becomes "gemini").
                if used_kind == "gemini" and status == 200 and out:
                    convert_pcm = True
            else:
                status, headers, out = _forward(
                    payload, self.command, upstream_path,
                    self.headers.get("Content-Type", "application/json"))

            resp_ctype = headers.get("Content-Type", "audio/mpeg")

            if convert_pcm and status == 200 and out:
                rate = _pcm_header_rate(headers.get("Content-Type", ""))
                try:
                    out = _pcm_to_mp3(out, rate)
                    resp_ctype = "audio/mpeg"
                except Exception as e:
                    out = json.dumps(
                        {"error": {"message": f"pcm->mp3 convert failed: {e}"}}).encode()
                    status = 502
                    resp_ctype = "application/json"

            # Peach 2026-08-31: full request log — capture model/length/status/
            # latency/kind so we can see EXACTLY what the app sent and got back,
            # especially for "Voice synthesis failed" errors on the app side.
            try:
                _dur = _time_mod.time() - _t0
                _req_model = _normalize_model(data.get("model", "")) if data else "?"
                _req_len = len(str(data.get("input", ""))) if data else 0
                _kind = "speech" if is_speech else "other"
                sys.stderr.write(
                    "[tts-proxy %s] %s model=%s inlen=%d status=%s kind=%s used=%s %dms ctype=%s%s\n" % (
                        _time_mod.strftime("%Y-%m-%d %H:%M:%S"),
                        "POST" if is_speech else self.command,
                        _req_model, _req_len, status,
                        _kind, used_kind,
                        int(_dur * 1000), resp_ctype,
                        (" BODY=" + str(out[:400].decode('utf-8', 'replace')) ) if status != 200 else ""))
            except Exception:
                pass

            self.send_response(status)
            self.send_header("Content-Type", resp_ctype)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            if self.command != "HEAD":
                try:
                    self.wfile.write(out)
                except (ConnectionResetError, BrokenPipeError):
                    # Client closed its side mid-write (e.g. Peach switched away
                    # from the voice page / app cancelled the request). Harmless —
                    # don't spam tracebacks, don't kill the thread handler.
                    sys.stderr.write("[tts-proxy] client reset during write (%s)\n" % self.path)
                except Exception as we:
                    sys.stderr.write("[tts-proxy] write error %s\n" % we)
        except Exception as e:
            try:
                body = json.dumps({"error": {"message": str(e)}}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception:
                pass


class V4Server(ThreadingHTTPServer):
    address_family = socket.AF_INET


class V6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def main():
    if not API_KEY:
        print("[tts-proxy] FATAL: OAUTH_OPENROUTER_KEY not set — refusing to start.")
        raise SystemExit(1)
    # Single dual-stack server (IPv4 loopback). This is the proven pattern from
    # the original marin proxy that answered 200 on 127.0.0.1:8765.
    # Binding two separate servers (v4+v6) caused the loopback to hang.
    servers = []
    try:
        s4 = ThreadingHTTPServer((HOST, PORT), Handler)
        servers.append(s4)
        print(f"[tts-proxy] OpenRouter TTS proxy v4 on http://{HOST}:{PORT} -> {UPSTREAM}")
    except OSError as e:
        print(f"[tts-proxy] v4 bind skipped: {e}")
    if HOST == "127.0.0.1":
        try:
            s6 = ThreadingHTTPServer(("::1", PORT), Handler)
            servers.append(s6)
            print(f"[tts-proxy] OpenRouter TTS proxy v6 on http://[::1]:{PORT} -> {UPSTREAM}")
        except OSError as e:
            print(f"[tts-proxy] v6 bind skipped: {e}")
    if not servers:
        raise SystemExit(1)
    for s in servers:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    print("[tts-proxy] listening")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        for s in servers:
            s.shutdown()


if __name__ == "__main__":
    main()