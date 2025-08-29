import pandas as pd

def run_pandas_code(code: str, df: pd.DataFrame):
    """Executes pandas code in a restricted namespace."""
    local_vars = {"df": df, "pd": pd}
    print('PANDAS CODE:')
    print(code)
    try:
        exec(code, local_vars)
        print('Finished executing code')
        # Expect user code to save result in variable 'result'
        return local_vars['result']
    except Exception as e:
        return f"Error executing code: {e}"
