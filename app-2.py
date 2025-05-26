import streamlit as st
import pandas as pd

st.title("AI-Powered Data Cleaner & Summarizer")
st.write("This app will clean your data and generate a summary.")
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])
if uploaded_file:
  df = pd.read_csv(uploaded_file)
  df.fillna(df.mean(numeric_only=True), inplace=True) 
  st.subheader("👀 Preview of Uploaded Data")
  st.dataframe(df.head())
  st.subheader("📊 Dataset Info")
  buffer = io.StringIO()
  df.info(buf=buffer)
  s = buffer.getvalue()
  st.text(s)

    # Display descriptive statistics
  st.subheader("📈 Summary Statistics")
  st.dataframe(df.describe())
