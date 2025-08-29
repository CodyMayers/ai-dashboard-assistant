import matplotlib.pyplot as plt

def render_plot(code: str, df):
    """Executes plotting code and returns a matplotlib figure."""
    fig, ax = plt.subplots()
    local_vars = {"df": df.copy(), "plt": plt, "ax": ax}
    try:
        exec(code, {"__builtins__": {}}, local_vars)
        return fig
    except Exception as e:
        return f"Error generating plot: {e}"
