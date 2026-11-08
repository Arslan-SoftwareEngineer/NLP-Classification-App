import streamlit as st
import requests

# Page configuration
st.set_page_config(page_title="NLP Classification App", page_icon="🤖")

st.title("🤖 NLP Classification App")
st.write("Test your fine-tuned BERT models for Sentiment Analysis and Topic Classification!")

# Configure Backend URL
# Update this URL once your FastAPI app is deployed on Render/Railway
BACKEND_URL = st.sidebar.text_input("Backend API URL:", value="http://127.0.0.1:8000/predict")

# Map user-friendly dropdown names to keys expected by FastAPI
task_options = {
    "SST-2 (Short Movie Sentiment)": "sst2",
    "IMDB (Long Movie Reviews)": "imdb",
    "AG News (News Topic Classification)": "ag_news"
}

selected_task_name = st.selectbox("Select a Task:", list(task_options.keys()))
selected_task_key = task_options[selected_task_name]

# Text input area
user_text = st.text_area("Enter text to analyze:", height=150)

# Submit button
if st.button("Analyze Text"):
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            payload = {
                "text": user_text,
                "task": selected_task_key
            }
            
            headers = {
                "ngrok-skip-browser-warning": "69420",
                "User-Agent": "StreamlitApp"
            }
            
            try:
                response = requests.post(
                    BACKEND_URL,
                    json=payload,
                    headers=headers,
                    timeout=15
                )
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        prediction = data.get("prediction", "N/A")
                        confidence = data.get("confidence", 0.0)
                        
                        st.success("Prediction Complete!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(label="Prediction", value=prediction)
                        with col2:
                            st.metric(label="Confidence Score", value=f"{confidence * 100:.2f}%")
                    except ValueError:
                        st.error("Server returned status 200, but response was not valid JSON.")
                else:
                    st.error(f"Backend Error (Status {response.status_code}): {response.text}")
                    
            except requests.exceptions.Timeout:
                st.error("Request timed out. Please check if your backend server is responsive.")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to backend. Verify your backend URL and server status.")