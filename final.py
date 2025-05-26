import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

st.set_page_config(page_title="LLaMA CSV Summarizer", layout="centered")
st.title("🦙 CSV Summarizer with LLaMA 3.2-1B")

# --- Step 1: User enters Hugging Face token ---
hf_token = st.text_input("🔐 Enter your Hugging Face token:", type="password")

if hf_token:
    try:
        login(token=hf_token)

        @st.cache_resource
        def load_model():
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=hf_token)
            model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", token=hf_token)
            return tokenizer, model

        tokenizer, model = load_model()
        st.success("✅ LLaMA model loaded successfully!")

        # --- Step 2: Upload CSV file ---
        uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.subheader("👀 Preview of Uploaded Data")
            st.dataframe(df.head())

            # --- Step 3: Generate Summary ---
            if st.button("🧠 Generate Summary"):
                with st.spinner("Running LLaMA model to generate summary..."):
                    def generate_summary(df):
                        rows = len(df)
                        cols = df.columns.tolist()
                        dtypes = df.dtypes.apply(str).to_dict()
                        missing = df.isnull().sum().to_dict()

                        prompt = f"""
                        Dataset Summary:
                        - Total rows: {rows}
                        - Columns: {cols}
                        - Data types: {dtypes}
                        - Missing values: {missing}

                        Please summarize this dataset in plain English.
                        """
                        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
                        outputs = model.generate(**inputs, max_new_tokens=200)
                        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        return decoded.split("Please summarize this dataset in plain English.")[-1].strip()

                    summary = generate_summary(df)
                st.subheader("📌 LLaMA Summary")
                st.write(summary)

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
else:
    st.info("🔑 Please enter your Hugging Face token to continue.")
