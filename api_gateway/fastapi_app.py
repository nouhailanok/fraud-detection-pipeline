from fastapi import FastAPI
import requests

app = FastAPI()

BENTO_URL = "http://localhost:3001"

# --- HEALTH (GET propre) ---
@app.get("/health")
def health():
    r = requests.post(f"{BENTO_URL}/health")
    return r.json()


# --- PREDICT proxy ---
@app.post("/predict")
def predict(payload: dict):
    r = requests.post(f"{BENTO_URL}/predict", json=payload)
    return r.json()


# --- METRICS ---
@app.get("/metrics")
def metrics():
    r = requests.post(f"{BENTO_URL}/metrics")
    return r.json()


# --- MODEL INFO ---
@app.get("/model/info")
def model_info():
    r = requests.post(f"{BENTO_URL}/model/info")
    return r.json()