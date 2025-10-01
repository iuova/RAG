import requests
import json

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "llama3:8b"

prompt = "Hi, say hello!"

payload = {
    "model": MODEL,
    "prompt": prompt,
    "stream": False,
    "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 128}
}

resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
resp.raise_for_status()
data = resp.json()
print("Ответ модели:", data.get("response"))
