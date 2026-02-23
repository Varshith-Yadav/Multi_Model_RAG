import requests

URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"


def generate(prompt: str, model: str = DEFAULT_MODEL) -> str:
    response = requests.post(
        URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=180,
    )
    response.raise_for_status()

    data = response.json()
    answer = data.get("response", "").strip()

    if not answer:
        raise RuntimeError(f"Empty response returned by Ollama for model '{model}'.")

    return answer
