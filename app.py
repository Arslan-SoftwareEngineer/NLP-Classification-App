import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configure Page Setup
st.set_page_config(page_title="NLP Classification App", page_icon="🤖", layout="centered")

st.title("🤖 NLP Classification App")
st.write("Analyze text using fine-tuned BERT models for Sentiment Analysis and News Topic Classification.")

# Define Hugging Face Repositories (Replace YOUR_HF_USERNAME with your actual account name)
MODEL_PATHS = {
    "SST-2 (Short Movie Sentiment)": ("sst2", "YOUR_HF_USERNAME/bert-sst2-classifier"),
    "IMDB (Long Movie Reviews)": ("imdb", "YOUR_HF_USERNAME/bert-imdb-classifier"),
    "AG News (News Topic Classification)": ("ag_news", "YOUR_HF_USERNAME/bert-agnews-classifier")
}

LABEL_MAPPINGS = {
    "sst2": {0: "Negative", 1: "Positive"},
    "imdb": {0: "Negative", 1: "Positive"},
    "ag_news": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
}

# Cache model and tokenizer in memory so they only load once
@st.cache_resource
def load_model_and_tokenizer(repo_id: str):
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    model.eval()
    return tokenizer, model

# User Selection UI
selected_option = st.selectbox("Select Classification Task:", list(MODEL_PATHS.keys()))
task_key, repo_id = MODEL_PATHS[selected_option]

user_text = st.text_area("Enter text to analyze:", height=160, placeholder="Type or paste your text here...")

if st.button("Analyze Text", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        with st.spinner("Fetching model and running inference..."):
            try:
                tokenizer, model = load_model_and_tokenizer(repo_id)
                inputs = tokenizer(user_text, return_tensors="pt", truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                    confidence, predicted_class = torch.max(probabilities, dim=-1)
                
                prediction = LABEL_MAPPINGS[task_key][predicted_class.item()]
                confidence_score = float(confidence.item())
                
                st.success("Analysis Complete!")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Predicted Class", value=prediction)
                with col2:
                    st.metric(label="Confidence Score", value=f"{confidence_score * 100:.2f}%")
                    
            except Exception as e:
                st.error(f"Error executing model inference: {str(e)}")