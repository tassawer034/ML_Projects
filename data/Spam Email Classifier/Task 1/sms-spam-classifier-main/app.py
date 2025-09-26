import os
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import streamlit as st

# --- Initialization ---
ps = PorterStemmer()

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in stopwords.words('english') and t not in string.punctuation]
    return " ".join(ps.stem(t) for t in tokens)


# --- Artifact Loader ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")
    model_path = os.path.join(BASE_DIR, "model.pkl")

    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        return None, None

    vectorizer = pickle.load(open(vectorizer_path, "rb"))
    model = pickle.load(open(model_path, "rb"))
    return vectorizer, model


# --- UI ---
st.set_page_config(
    page_title="Spam Classifier",
    layout="centered",
    page_icon="📧"
)

# --- Custom CSS for color theme ---
st.markdown(
    """
    <style>
    /* Background */
    .stApp {
        background-color: #f4f7fa;
    }
    /* Title */
    .main-title {
        color: #1e3a8a;
        font-size: 2.2rem;
        font-weight: bold;
    }
    /* Author name */
    .author-name {
        text-align: right;
        font-weight: bold;
        color: #2563eb;
    }
    /* Buttons */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 10px;
        padding: 8px 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1e40af;
        color: #f9fafb;
    }
    /* Text area */
    textarea {
        border: 2px solid #2563eb !important;
        border-radius: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([6, 1])
with col1:
    st.markdown("<p class='main-title'>📧 Email / SMS Spam Classifier</p>", unsafe_allow_html=True)
with col2:
    st.markdown("<p class='author-name'>Tasawar Ali</p>", unsafe_allow_html=True)

st.write("Enter a message below and check whether it's Spam or Not Spam.")

input_sms = st.text_area("✍️ Enter your message here")

vectorizer, model = load_artifacts()

if vectorizer is None or model is None:
    st.error("❌ Model files not found! Please run `train_model.py` first to generate `vectorizer.pkl` and `model.pkl`.")
else:
    if st.button("🔍 Predict"):
        transformed_sms = transform_text(input_sms)
        vector_input = vectorizer.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 Spam Detected!")
        else:
            st.success("✅ This is Not Spam.")
