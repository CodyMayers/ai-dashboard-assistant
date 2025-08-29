import streamlit as st
import pandas as pd
from agent_setup import setup_agent
import os
from joblib import load, dump

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("AI-Powered Dashboard Assistant")

# Upload CSV
if os.listdir('cache'):
    print('loading from cache')
    df = load('cache/df.joblib')
else:
    print('loading from csv')
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    df = pd.read_csv('data/rejected_2007_to_2018Q4.csv')
    dump(df, 'cache/df.joblib')

if df is not None:
    st.write("Preview of your data:")
    st.dataframe(df.head())
    st.write("Columns:", df.columns.tolist())

    # Text input for multi-turn questions
    user_question = st.text_input("Ask a question about your data:")

    if "agent" not in st.session_state:
        agent = setup_agent(df)
        st.session_state.agent = agent
        st.session_state.df = df

    # Run agent on user input
    if user_question:
        config = {"configurable": {"thread_id": "abc123"}}
        input_message = {
            "role": "user",
            "content": user_question,
        }
        for step in st.session_state.agent.stream(
            {"messages": [input_message]},
            stream_mode='values',
            config=config
        ):
            got_result = False
            for msg in step['messages']:
                # if the msg is a ToolMessage, get the value of the content attribute
                if msg.type == 'tool':
                    result = msg.content
                    got_result = True
                    print(result)
                    break
            if got_result:
                break
        print('GOT RESULT: ---------------------------------------------------------------')
        print(result)
        print('PRINTED RESULT ------------------------------------------------------------')
        # Display result: table, plot, or text
        if isinstance(result, pd.DataFrame):
            st.dataframe(result)
        elif hasattr(result, "show"):  # matplotlib figure
            st.pyplot(result)
        else:
            st.write(result)
