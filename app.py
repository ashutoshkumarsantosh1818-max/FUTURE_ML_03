import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="CUSTOMER SUPPORT AI", page_icon="🤖")

# --- DATA LOADING ENGINE ---
@st.cache_resource
def initialize_engine():
    # Use simple filename since it's in the same folder
    file_name = 'dialogs.csv'
    
    if not os.path.exists(file_name):
        return None, None, None, f"Error: {file_name} not found in folder."
    
    try:
        # Step 1: Read file line by line and parse manually (handles quoted lines with tabs)
        data = []
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Remove outer quotes if present
                if line.startswith('"') and line.endswith('"'):
                    line = line[1:-1]
                # Split by tab
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    question = parts[0].strip().strip('"').strip("'")
                    answer = parts[1].strip().strip('"').strip("'")
                    if question and answer:
                        data.append({'question': question, 'answer': answer})
        
        df = pd.DataFrame(data)
        
        if df.empty:
            # Fallback: Try standard CSV reading
            df = pd.read_csv(file_name, sep='\t', names=['question', 'answer'], encoding='utf-8', on_bad_lines='skip')
            if df.shape[1] < 2:
                df = pd.read_csv(file_name, sep=',', names=['question', 'answer'], encoding='utf-8', on_bad_lines='skip')
            
            # Clean the data - remove quotes if present
            df['question'] = df['question'].astype(str).str.strip().str.strip('"').str.strip("'")
            df['answer'] = df['answer'].astype(str).str.strip().str.strip('"').str.strip("'")
        
        # Remove rows where question or answer is empty, NaN, or just whitespace
        df = df[df['question'].notna() & df['answer'].notna()]
        df = df[(df['question'].str.len() > 0) & (df['answer'].str.len() > 0)]
        df = df[df['question'] != 'nan']  # Remove rows where question is literally "nan"
        df = df[df['answer'] != 'nan']    # Remove rows where answer is literally "nan"
        df = df.dropna().drop_duplicates().reset_index(drop=True)
        
        # Check if we actually have data
        if df.empty:
            return None, None, None, "Error: The file is empty or contains no valid data."

        # Step 2: Create TF-IDF Matrix
        # 'stop_words' hata diya hai taaki chote words par 'empty vocabulary' error na aaye
        vectorizer = TfidfVectorizer(lowercase=True)
        tfidf_matrix = vectorizer.fit_transform(df['question'].values.astype('U'))
        
        return df, vectorizer, tfidf_matrix, "Success"
    except Exception as e:
        return None, None, None, f"Critical Error: {str(e)}"

# Initialize
df, vectorizer, tfidf_matrix, status = initialize_engine()

# --- UI DESIGN ---
st.title("🤖 NARAYAN AI")
st.caption("Internship Project: Intent Recognition & Smart Fallback")

if status != "Success":
    st.error(status)
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # NLP Processing
        query_vec = vectorizer.transform([prompt.lower()])
        similarity_scores = cosine_similarity(query_vec, tfidf_matrix)
        
        if similarity_scores.max() < 0.2:
            response = "I'm sorry, I don't have an answer for that. Can you rephrase?"
        else:
            best_match_idx = similarity_scores.argmax()
            response = df['answer'].iloc[best_match_idx]
            
            # Check if response is NaN or invalid
            if pd.isna(response) or str(response).strip().lower() == 'nan' or len(str(response).strip()) == 0:
                response = "I'm sorry, I don't have an answer for that. Can you rephrase?"

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})