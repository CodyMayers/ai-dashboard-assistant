import pandas as pd
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent


class DataAnalyst():
    def __init__(self, conversation_id: str):
        llm = ChatOllama(model="gpt-oss:20b")
        memory = InMemorySaver()
        self.agent = create_react_agent(
            model=llm, 
            tools=[], 
            checkpointer=memory
        )
        self.config = {"configurable": {"thread_id": conversation_id}}

    def _execute_pandas_code(self, df: pd.DataFrame, query: str):
        print('Executing query:', query)
        local_vars = {"df": df, "pd": pd} # things that exec() will need access to in its limited scope
        exec(query, {}, local_vars)
        return local_vars['result']

    def query(self, df: pd.DataFrame, message: str):
        # We don't want this as a prompt when the agent is first created because the df could change between questions
        system_message = {
            "role": "system",
            "content": (
                "You are a data analyst's assistant. The dataset is in a Pandas DataFrame with the following columns: "
                f"{', '.join(df.columns)}.\n"
                "Generate Python code using pandas to either analyze or manipulate the DataFrame based on the user's request. "
                "In your code, do not import anything or define the DataFrame. "
                "Always tart your code with `result = ` followed by the pandas code. "
                "Do not add explanations, markdown code fences, or any text other than the code."
            )
        }
        user_message = {"role": "user", "content": message}
        result = self.agent.invoke(
            {"messages": [system_message, user_message]},
            self.config
        )
        code = result['messages'][-1].content
        return self._execute_pandas_code(df, code)
