import requests

URL = "http://localhost:11434/api/embeddings"

def embed(text):
    res = requests.post(
        URL,
        json={
            "model": "nomic-embed-text",
            "input": text
        }
    )
    return res.json()['embedding']

