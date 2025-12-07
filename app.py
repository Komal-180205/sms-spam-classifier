import streamlit as st
import pickle
import re
from typing import Any, Tuple

# --- 1. SET PAGE CONFIGURATION ---
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="wide")

# --- CUSTOM CSS FOR UI DESIGN ---
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: white;
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(135deg, #1E88E5, #4CAF50);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    margin-bottom: 30px;
}

.sub-header {
    text-align: center;
    font-size: 1.1em;
    color: #455A64;
    margin-bottom: 40px;
}

textarea {
    border-radius: 12px !important;
    border: 1px solid #bbb !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 12px;
    font-weight: bold;
    padding: 10px 20px;
    transition: 0.3s;
    border: none;
}

.stButton>button:hover {
    color: white;
    background-color: #43A047;
}

.result-box-spam {
    color : black;
    border: 2px solid #00000;
    background-color: #FFEBEE;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    font-size: 1.15em;
    font-weight: bold;
}

.result-box-notspam {
    color : black;
    border: 2px solid #00000;
    background-color: #E8F5E9;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    font-size: 1.15em;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --- TEXT PREPROCESSING ---
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- LOAD MODEL & VECTORIZER ---
@st.cache_resource
def load_assets() -> Tuple[Any, Any]:
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model/vectorizer: {e}")
        st.stop()

model, vectorizer = load_assets()

# --- HEADER ---
st.markdown("""
    <div class="main-header">
        <h1>📩 SMS Spam Classifier</h1>
    </div>
    <p class="sub-header">
        Paste any SMS message and let AI instantly detect whether it's <b>Spam</b> or <b>Not Spam</b>.
    </p>
""", unsafe_allow_html=True)

# --- UI LAYOUT ---
col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    st.subheader("Enter Message:")
    user_input = st.text_area(
        "Paste SMS here:",
        height=180,
        placeholder="Example: Congratulations! You won a ₹5000 gift card. Click the link to claim.",
        label_visibility="collapsed"
    )

    st.markdown("---")

    if st.button("🚀 Analyze Message", use_container_width=True):
        if not user_input.strip():
            st.warning("Please enter a message to analyze!")
        else:
            # Preprocess
            cleaned_input = preprocess_text(user_input)

            # Vectorize
            X_sparse = vectorizer.transform([cleaned_input])
            X_dense = X_sparse.toarray()

            # Predict
            prediction = model.predict(X_dense)[0]

            st.markdown("---")
            st.subheader("Analysis Result")

            # SPAM RESULT
            if prediction == 1:
                st.markdown(
                    """
                    <div class="result-box-spam">
                        🚨 <b>SPAM MESSAGE DETECTED!</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.info("⚠️ This message shows strong indicators of spam. Avoid clicking links or replying.")

            # NOT SPAM RESULT
            else:
                st.markdown(
                    """
                    <div class="result-box-notspam">
                        ✅ <b>Message is NOT SPAM</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.success("🎉 This message appears to be safe and legitimate.")

# --- FOOTER ---
st.markdown("""
---
<footer style='text-align: center; color: #777; font-size: 0.8em; margin: 0 0 0 0; padding :1px 0 1px 0;'>
    Designed by team with ❤️ <br>
Intelligent SMS Security
</footer>
""", unsafe_allow_html=True)
