import os
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")

def ollama_generate(prompt: str) -> str:
    """
    Calls Ollama's /api/generate endpoint and returns the generated text.
    Uses a simple non-streaming request for clarity.
    """
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()
