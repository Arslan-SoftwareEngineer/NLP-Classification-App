import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="NLP Classification API")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Label Mappings
LABEL_MAPPINGS = {
    "sst2": {0: "Negative", 1: "Positive"},
    "imdb": {0: "Negative", 1: "Positive"},
    "ag_news": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
}

# Model paths (local folder paths)
MODEL_PATHS = {
    "sst2": "./bert_sst2_final",
    "imdb": "./bert_imdb",
    "ag_news": "./bert_agnews"
}

# Cache for lazy loading models in memory
models = {}
tokenizers = {}

class PredictionRequest(BaseModel):
    text: str
    task: str

def load_model_and_tokenizer(task: str):
    """Lazy load models on demand to save memory."""
    if task not in models:
        model_path = MODEL_PATHS.get(task)
        if not model_path or not os.path.exists(model_path):
            raise HTTPException(
                status_code=404, 
                detail=f"Model directory for task '{task}' not found on server."
            )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        
        tokenizers[task] = tokenizer
        models[task] = model

    return tokenizers[task], models[task]

@app.get("/")
def root():
    return {"status": "online", "message": "NLP Classification Backend is running."}

@app.post("/predict")
def predict(request: PredictionRequest):
    task = request.task.lower()
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    if task not in LABEL_MAPPINGS:
        raise HTTPException(status_code=400, detail=f"Invalid task '{task}'. Valid tasks: sst2, imdb, ag_news")

    tokenizer, model = load_model_and_tokenizer(task)

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        confidence, predicted_class = torch.max(probabilities, dim=-1)

    predicted_label = LABEL_MAPPINGS[task][predicted_class.item()]
    confidence_score = float(confidence.item())

    return {
        "task": task,
        "prediction": predicted_label,
        "confidence": confidence_score
    }