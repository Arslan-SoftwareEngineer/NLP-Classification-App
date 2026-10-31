import streamlit as st
import requests

# Page configuration
st.set_page_config(page_title="NLP Classification App", page_icon="🤖")

st.title("🤖 NLP Classification App")
st.write("Test your fine-tuned BERT models for Sentiment Analysis and Topic Classification!")

# Map user-friendly dropdown names to the exact keys your FastAPI backend expects
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
            # Prepare the data to send to FastAPI
            payload = {
                "text": user_text,
                "task": selected_task_key
            }
            
            try:
                # Send a POST request to your local FastAPI server
                response = requests.post("https://nlp-classification-app.streamlit.app/predict", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    prediction = data["prediction"]
                    confidence = data["confidence"]
                    
                    st.success("Prediction Complete!")
                    
                    # Display the results nicely
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Prediction", value=prediction)
                    with col2:
                        st.metric(label="Confidence Score", value=f"{confidence * 100:.2f}%")
                else:
                    st.error(f"Error from server: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the backend. Is your FastAPI server running?")