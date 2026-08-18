from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="NLP Classification API")

# Define the expected request body
class TextRequest(BaseModel):
    text: str
    task: str  # 'sst2', 'imdb', or 'ag_news'

# Lazy-load models to save your laptop's memory
model_pipelines = {}

def get_pipeline(task: str):
    if task not in model_pipelines:
        try:
            if task == "sst2":
                model_pipelines[task] = pipeline("sentiment-analysis", model="./bert_sst2_final", tokenizer="bert-base-uncased")
            elif task == "imdb":
                model_pipelines[task] = pipeline("sentiment-analysis", model="./bert_imdb", tokenizer="bert-base-uncased")
            elif task == "ag_news":
                model_pipelines[task] = pipeline("text-classification", model="./bert_agnews", tokenizer="bert-base-uncased")
            else:
                raise ValueError("Invalid task")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
    return model_pipelines[task]

# Label decoders
def decode_label(task: str, label: str) -> str:
    if task in ["sst2", "imdb"]:
        return "Positive" if label == "LABEL_1" else "Negative"
    elif task == "ag_news":
        mapping = {"LABEL_0": "World", "LABEL_1": "Sports", "LABEL_2": "Business", "LABEL_3": "Sci/Tech"}
        return mapping.get(label, label)

@app.post("/predict")
async def predict(request: TextRequest):
    if request.task not in ["sst2", "imdb", "ag_news"]:
        raise HTTPException(status_code=400, detail="Task must be sst2, imdb, or ag_news")
    
    classifier = get_pipeline(request.task)
    result = classifier(request.text)[0]
    
    return {
        "text": request.text,
        "task": request.task,
        "prediction": decode_label(request.task, result["label"]),
        "confidence": round(result["score"], 4)
    }