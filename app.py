import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Page Config
st.set_page_config(page_title="NLP-BERT Classifier Hub", page_icon="⚡", layout="centered")

# Custom Dark Styling
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e6ed; }
    .header-card {
        background: linear-gradient(135deg, #1e2640 0%, #111827 100%);
        border: 1px solid #2e3856;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .header-title { font-size: 28px; font-weight: 700; color: #ffffff; margin: 0 0 8px 0; }
    .header-sub { color: #94a3b8; font-size: 14px; margin: 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-card">
    <h1 class="header-title">⚡ Deep Learning NLP Classifier</h1>
    <p class="header-sub">Fine-tuned BERT models for real-time text classification & sentiment analysis.</p>
</div>
""", unsafe_allow_html=True)

# Replace YOUR_HF_USERNAME with your Hugging Face username
MODEL_PATHS = {
    "SST-2 (Short Movie Sentiment)": ("sst2", "ARS3LAN/bert-sst2-classifier"),
    "IMDB (Long Movie Reviews)": ("imdb", "ARS3LAN/bert-imdb-classifier"),
    "AG News (News Topic Classification)": ("ag_news", "ARS3LAN/bert-agnews-classifier")
}

LABEL_MAPPINGS = {
    "sst2": {0: "Negative", 1: "Positive"},
    "imdb": {0: "Negative", 1: "Positive"},
    "ag_news": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
}

@st.cache_resource
def load_model_and_tokenizer(repo_id: str):
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    model.eval()
    return tokenizer, model

with st.sidebar:
    st.markdown("### 📊 Task Details")
    st.markdown("""
    - **SST-2**: Short phrase sentiment
    - **IMDB**: Full movie review sentiment
    - **AG News**: Multi-class news topics
    """)
    st.caption("Powered by PyTorch & Hugging Face")

selected_option = st.selectbox("🎯 Select Task Model", list(MODEL_PATHS.keys()))
task_key, repo_id = MODEL_PATHS[selected_option]

user_text = st.text_area("📝 Input Text for Classification", height=150, placeholder="Type or paste text here...")

if st.button("Analyze Text", use_container_width=True, type="primary"):
    if not user_text.strip():
        st.warning("⚠️ Please provide text input before running analysis.")
    else:
        with st.spinner("Loading model from Hugging Face & running inference..."):
            try:
                tokenizer, model = load_model_and_tokenizer(repo_id)
                inputs = tokenizer(user_text, return_tensors="pt", truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                    confidence, predicted_class = torch.max(probabilities, dim=-1)
                
                prediction = LABEL_MAPPINGS[task_key][predicted_class.item()]
                score = float(confidence.item())
                
                st.markdown("---")
                st.subheader("📊 Inference Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Label", prediction)
                with col2:
                    st.metric("Confidence Score", f"{score * 100:.2f}%")
                    st.progress(score)
            except Exception as e:
                st.error(f"Error loading model or evaluating text: {str(e)}")