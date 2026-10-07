import requests

from .config import OLLAMA_URL, OLLAMA_MODEL


def generate(prompt: str, model: str = OLLAMA_MODEL) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "seed": 42,
            },
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["response"].strip()