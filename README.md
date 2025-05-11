# 💬 DialoGPT Chatbot (Streamlit)

This is a simple chatbot interface built using [Streamlit](https://streamlit.io/) and [DialoGPT-medium](https://huggingface.co/microsoft/DialoGPT-medium) from Hugging Face Transformers.

---

## 🖥️ Setup Instructions

You can run this chatbot on **Windows** or **Linux**. Follow the steps below to set it up.

---

## ✅ Prerequisites

- Python 3.8 or higher
- Git (optional but recommended)
- Internet connection (for downloading model)

---

## 🐍 Create a Virtual Environment

### 🔷 On Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
````

### 🟦 On Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 📦 Install Requirements

```bash
pip install -r streamlit transformers torch
```

---

## 🚀 Run the Chatbot

```bash
streamlit run app.py
```

Then open the browser link (usually [http://localhost:8501](http://localhost:8501)) to chat!

---

## 💡 Features

* Multi-turn dialogue using DialoGPT
* Clean Streamlit interface
* Session-based memory to hold chat context

---

## 🧼 Deactivate Environment

When you're done:

```bash
deactivate
```
