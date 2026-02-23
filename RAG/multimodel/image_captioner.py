import base64
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llava"
DEFAULT_PROMPT = "Describe the chart or diagram in detail."


def caption_image(path: str, model: str = DEFAULT_MODEL, prompt: str = DEFAULT_PROMPT) -> str:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    encoded_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "images": [encoded_image],
            "stream": False,
        },
        timeout=180,
    )
    response.raise_for_status()

    data = response.json()
    return data.get("response", "").strip()

