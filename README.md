# 🤖 NLP Transformer Classification Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-F9AB00?style=for-the-badge&logo=huggingface)

An end-to-end Natural Language Processing pipeline featuring fine-tuned BERT models for both sentiment analysis and multi-class topic classification. This project integrates a FastAPI backend for efficient model inference and a Streamlit web interface for real-time user interaction.

Developed as a core project during the Deep Learning Internship at **Ezitech Institute**.

---

## ✨ Features

* **Multi-Task Classification:** Includes models fine-tuned on three distinct datasets:
  * **SST-2:** Binary sentiment analysis on short movie reviews.
  * **IMDB:** Binary sentiment analysis on long-form movie reviews.
  * **AG News:** 4-way topic classification (World, Sports, Business, Sci/Tech).
* **RESTful API Backend:** Built with FastAPI to lazy-load models, process requests, and return confidence scores instantly.
* **Interactive Web Frontend:** Built with Streamlit to provide a seamless, user-friendly dashboard for text analysis.
* **Cloud-Trained:** Models fine-tuned using Google Colab T4 GPUs and the Hugging Face `transformers` library.

---

## 🛠️ Tech Stack

* **Machine Learning:** PyTorch, Hugging Face Transformers, Datasets, Evaluate
* **Backend:** FastAPI, Uvicorn, Pydantic
* **Frontend:** Streamlit, Requests
* **Environment:** Ubuntu (Xubuntu/Xfce), Python 3.12

---

## 📂 Project Structure

```text
├── main.py                  # FastAPI backend server
├── frontend.py              # Streamlit web interface
├── .gitignore               # Ignored heavy model files
├── bert_sst2_final/         # Fine-tuned SST-2 model (Not pushed to GitHub)
├── bert_imdb/               # Fine-tuned IMDB model (Not pushed to GitHub)
└── bert_agnews/             # Fine-tuned AG News model (Not pushed to GitHub)
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/Arslan-SoftwareEngineer/NLP-Classification-App.git](https://github.com/Arslan-SoftwareEngineer/NLP-Classification-App.git)
cd NLP-Classification-App
```

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note: Ensure you have downloaded the fine-tuned model folders into the root directory before running).*

### 4. Run the Application

You need two terminal windows to run both the backend and frontend simultaneously.

**Terminal 1: Start the FastAPI Backend**
```bash
uvicorn main:app --reload
```
The API will be available at `http://127.0.0.1:8000/docs`

**Terminal 2: Start the Streamlit Frontend**
```bash
streamlit run frontend.py
```
The web app will automatically open at `http://localhost:8501`

## 💡 Usage

1. Open the Streamlit web interface.
2. Select your desired classification task from the dropdown menu.
3. Enter or paste the text you want to analyze.
4. Click **"Analyze Text"** to receive the predicted label and the model's confidence score.
