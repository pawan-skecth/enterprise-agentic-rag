import base64
import time
from io import BytesIO
from PIL import Image
from groq import Groq

import config


# Initialize Groq client once
_client = Groq(api_key=config.GROQ_API_KEY)

# Prompt
_VLM_PROMPT = (
    "Describe this image factually and concisely in 2-4 sentences. "
    "Focus on: what the image shows, any text labels visible, "
    "and the key information it conveys. "
    "If it is a chart or diagram, describe the structure and main data points. "
    "Do NOT invent details — only describe what you actually see."
)

# ── Rate limiting ─────────────────────────────────
# Groq free tier: ~7000 input tokens/minute
# Each VLM call ≈ 2000 tokens → space calls ≥4s apart
_MIN_INTERVAL_SEC = 4.0
_last_call_time = 0.0

# ── Retry ─────────────────────────────────────────
_MAX_RETRIES = 4
_BASE_WAIT = 3.0   # initial wait on 429; doubles each retry


def _wait_for_slot():
    """Block until enough time has passed since the last VLM call."""
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < _MIN_INTERVAL_SEC:
        time.sleep(_MIN_INTERVAL_SEC - elapsed)
    _last_call_time = time.time()


def describe_image(image_input, max_tokens: int = 300) -> str:
    """
    Send image to Groq VLM with rate limiting + retry. Returns '' on failure.
    """
    # Load image bytes
    if isinstance(image_input, str):
        with open(image_input, "rb") as f:
            img_bytes = f.read()
    elif isinstance(image_input, bytes):
        img_bytes = image_input
    else:
        raise TypeError("image_input must be a path (str) or bytes")

    # Re-encode as PNG
    img = Image.open(BytesIO(img_bytes))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    # Retry loop with exponential backoff
    for attempt in range(_MAX_RETRIES):
        _wait_for_slot()
        try:
            response = _client.chat.completions.create(
                model=config.GROQ_VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _VLM_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e)
            is_rate_limit = "429" in msg or "rate_limit" in msg.lower()
            if is_rate_limit and attempt < _MAX_RETRIES - 1:
                wait = _BASE_WAIT * (2 ** attempt)
                print(f"[VLM] Rate limit — waiting {wait:.1f}s (retry {attempt+1}/{_MAX_RETRIES})")
                time.sleep(wait)
                continue
            print(f"[VLM] Error: {e}")
            return ""

    return ""