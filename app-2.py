import streamlit as st
import pandas as pd

st.title("AI-Powered Data Cleaner & Summarizer")
st.write("This app will clean your data and generate a summary.")
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])
if uploaded_file:
  df = pd.read_csv(uploaded_file)
  st.subheader("👀 Preview of Uploaded Data")
  st.dataframe(df.head())
