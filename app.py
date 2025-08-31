import streamlit as st
import pandas as pd
import os
from joblib import load, dump
from data_analyst import DataAnalyst

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("AI-Powered Dashboard Assistant")

# Upload CSV
df = None
if os.listdir('cache'):
    print('loading from cache')
    df = load('cache/df.joblib')
else:
    print('loading from csv')
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        dump(df, 'cache/df.joblib')

if df is not None:
    st.write("Preview of your data:")
    st.dataframe(df.head())
    # st.write("Columns:", df.columns.tolist())

    # Text input for multi-turn questions
    user_question = st.text_input("Ask a question about your data:")

    if "agent" not in st.session_state:
        analyst = DataAnalyst(conversation_id="abc123")
        st.session_state.analyst = analyst
        st.session_state.df = df

    # Run agent on user input
    if user_question:
        result = st.session_state.analyst.query(st.session_state.df, user_question)
        # Display result: table, plot, or text
        if isinstance(result, pd.DataFrame):
            st.dataframe(result)
        elif hasattr(result, 'figure'):
            st.pyplot(result.figure)
        else:
            st.write(result)
