from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState
import pandas as pd
from joblib import load, dump
import os

if os.listdir('cache'):
    print('loading from cache')
    df = load('cache/df.joblib')
else:
    print('loading from csv')
    df = pd.read_csv('data/rejected_2007_to_2018Q4.csv')
    dump(df, 'cache/df.joblib')
print('Loaded data')

df = df.head()

# Tool function
def run_pandas_code(code: str, df: pd.DataFrame):
    """Executes pandas code in a restricted namespace."""
    print('run_pandas_code')
    local_vars = {"df": df, "pd": pd}
    try:
        exec(code, local_vars)
        print('Finished executing code')
        # Expect user code to save result in variable 'result'
        return local_vars['result']
    except Exception as e:
        return f"Error executing code: {e}"

# Custom prompt to only return tool output
def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:  
    print('Creating prompt')
    df_columns = config["configurable"].get("df_columns")
    print(df_columns)
    system_msg = (
        f"You are a Python data assistant. The DataFrame is called `df` with columns: {df_columns}.\n"
        "Answer the user's request. "
        "Always respond with Python code ONLY. Do not add explanations. "
        "If it involves data analysis, generate Python pandas code ONLY. "
        "If it involves plotting, generate matplotlib or plotly code. "
        "Assign results to a variable named `result`."
        "Do not include markdown code fences. Do NOT return text explanations with code."
    )
    return [{"role": "system", "content": system_msg}] + state["messages"]

# Conversation memory for multi-turn context
# memory = MemorySaver()

# Create the model and agent
llm = ChatOllama(model="mistral:latest", temperature=0)
agent = create_react_agent(
    model=llm,
    tools=[run_pandas_code],
    prompt=prompt,
)

# Define the inputs
config = {"configurable": {"thread_id": "abc123", "df_columns": list(df.columns)}}
input_message = {
    "role": "user",
    "content": "What is the average of the Risk_Score column?",
}

# Run the agent
for step in agent.stream(
    {"messages": [input_message]},
    stream_mode='values'
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
        

# from agent_setup import setup_agent


# agent = setup_agent(df)

# # Use the agent
# config = {"configurable": {"df_columns": list(df.columns), "thread_id": "abc123"}}

# question = "What is the average 'Risk_Score' in the data?"

# # Run the agent
# response = agent.invoke(
#     {"messages": [{"role": "user", "content": question}]},
#     config
# )

# print(response)