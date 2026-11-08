import streamlit as st
import requests

# Page Config
st.set_page_config(
    page_title="BERT Classifier Hub",
    page_icon="⚡",
    layout="centered"
)

# Custom High-Contrast UI Styling
st.markdown("""
<style>
    /* Dark theme adjustments */
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    
    /* Header Container */
    .header-card {
        background: linear-gradient(135deg, #1e2640 0%, #111827 100%);
        border: 1px solid #2e3856;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .header-title {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 8px 0;
    }
    
    .header-sub {
        color: #94a3b8;
        font-size: 14px;
        margin: 0;
    }

    /* Output Card */
    .result-card {
        background-color: #1a2234;
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
    }

    .badge {
        display: inline-block;
        padding: 6px 14px;
        font-weight: 600;
        border-radius: 20px;
        background: #2563eb;
        color: #ffffff;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Custom Header UI
st.markdown("""
<div class="header-card">
    <h1 class="header-title">⚡ Deep Learning NLP Classifier</h1>
    <p class="header-sub">Fine-tuned BERT models for real-time text classification & sentiment analysis.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Settings & About
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Hide raw URL field behind an expander or set intelligent fallback
    with st.expander("🌐 Advanced API Endpoint"):
        backend_url = st.text_input(
            "Backend URL", 
            value="http://127.0.0.1:8000/predict",
            help="Default points to local FastAPI service. Update if using external host."
        )

    st.divider()
    st.markdown("### 📊 Models Info")
    st.markdown("""
    - **SST-2**: Binary sentiment on short phrases
    - **IMDB**: Detailed movie review sentiment
    - **AG News**: World, Sports, Business, Sci/Tech
    """)
    st.caption("Powered by PyTorch & Transformers")

# Model Selection mapping
TASK_MAP = {
    "SST-2 (Short Movie Sentiment)": "sst2",
    "IMDB (Long Movie Reviews)": "imdb",
    "AG News (News Topic Classification)": "ag_news"
}

# Main Selection & Input
selected_label = st.selectbox("🎯 Select Task Model", list(TASK_MAP.keys()))
task_code = TASK_MAP[selected_label]

user_text = st.text_area(
    "📝 Input Text for Classification",
    height=150,
    placeholder="Type or paste text here to analyze..."
)

# Trigger Analysis
if st.button("🚀 Analyze Text", use_container_width=True, type="primary"):
    if not user_text.strip():
        st.warning("⚠️ Please provide text input before running analysis.")
    else:
        with st.spinner("Processing through BERT pipeline..."):
            try:
                payload = {"text": user_text, "task": task_code}
                response = requests.post(backend_url, json=payload, timeout=10)
                
                if response.status_code == 200:
                    res_data = response.json()
                    pred = res_data.get("prediction", "N/A")
                    conf = res_data.get("confidence", 0.0)
                    
                    st.markdown("---")
                    st.subheader("📊 Inference Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Label", pred)
                    with col2:
                        st.metric("Confidence Score", f"{conf * 100:.2f}%")
                        st.progress(conf)
                else:
                    st.error(f"Backend API returned status code: {response.status_code}")
            except Exception as e:
                st.error(f"Could not connect to API server at `{backend_url}`. Ensure `main.py` is running.")