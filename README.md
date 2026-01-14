# 🤖 Narayan AI - Smart Intent Recognition Chatbot

Narayan AI is an intelligent customer support solution developed for Internship Task 3. The bot is designed to understand user queries and provide automated responses using Natural Language Processing (NLP) techniques.

## 🚀 Live Demo
**Link:** [PASTE_YOUR_STREAMLIT_LINK_HERE]

## ✨ Key Features
* **Branding:** The application is titled **Narayan AI** with a browser page title set to **Customer Support AI**.
* **Intent Recognition:** Utilizes TF-IDF Vectorization to analyze and match user intents with high accuracy.
* **Smart Fallback Mechanism:** If a user's query does not match the dataset (similarity < 20%), the bot provides a polite automated fallback response.
* **Large Knowledge Base:** Powered by a `dialogs.csv` dataset containing over 3,500+ conversation pairs.

## 🛠️ Tech Stack
* **Frontend:** Streamlit (Web Interface)
* **Language:** Python 3.10+
* **Machine Learning:** Scikit-Learn (TF-IDF, Cosine Similarity)
* **Data Management:** Pandas

## 📂 Project Structure
```text
├── app.py              # Core application logic and Narayan AI interface
├── dialogs.csv         # Knowledge base dataset
├── requirements.txt    # Dependency list (streamlit, pandas, scikit-learn)
└── README.md           # Project documentation and setup guide
