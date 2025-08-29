import pandas as pd
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.agents import Tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState
from analysis_executor import run_pandas_code
from plotting_tool import render_plot


def setup_agent(df: pd.DataFrame):
    # -----------------------------
    # LangChain Setup
    # -----------------------------
    def pandas_executor_wrapper(code):
        print('---GOT PANDAS CODE-----------------------')
        print(code)
        if not code.startswith('result = '):
            code = "result = " + code
        print(code)
        return run_pandas_code(code, df)

    def plot_generator_wrapper(code):
        return render_plot(code, df)
    
    def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:  
        df_columns = config["configurable"].get("df_columns")
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
    memory = MemorySaver()
    
    # Define tools
    tools = [
        Tool(
            name="PandasExecutor",
            func=pandas_executor_wrapper,
            description="Executes pandas code on the DataFrame and returns results."
        ),
        Tool(
            name="PlotGenerator",
            func=plot_generator_wrapper,
            description="Generates matplotlib plots and returns figure objects."
        ),
    ]

    # Initialize the Mistral model via Ollama API wrapper
    llm = ChatOllama(model="mistral:latest", temperature=0)

    # Initialize agent with multi-tool capability
    agent_executor = create_react_agent(llm, tools, checkpointer=memory, prompt=prompt)

    return agent_executor