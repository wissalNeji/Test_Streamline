import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# --- Authenticate with Hugging Face (token stored in Streamlit secrets) ---
login(token=st.secrets["huggingface"]["token"])

# --- Load LLaMA Model and Tokenizer ---
@st.cache_resource
def load_llama_model():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    return tokenizer, model

tokenizer, model = load_llama_model()

# --- Summarization Logic ---
def generate_summary(df):
    # Basic profiling
    rows = len(df)
    cols = df.columns.tolist()
    dtypes = df.dtypes.to_dict()
    missing = df.isnull().sum().to_dict()

    # Construct a prompt
    prompt = f"""
    Dataset Summary:
    - Total rows: {rows}
    - Columns: {cols}
    - Data types: {dtypes}
    - Missing values: {missing}

    Please summarize this dataset in plain English.
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract response after prompt
    return decoded.split("Please summarize this dataset in plain English.")[-1].strip()

# --- Streamlit App ---
st.set_page_config(page_title="LLaMA CSV Summarizer", layout="centered")
st.title("📊 CSV Summarizer with LLaMA 3.2-1B")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("👀 Preview of Data")
    st.dataframe(df.head())

    if st.button("🧠 Generate Summary"):
        with st.spinner("Running LLaMA model..."):
            summary = generate_summary(df)
        st.subheader("📌 Dataset Summary")
        st.write(summary)
