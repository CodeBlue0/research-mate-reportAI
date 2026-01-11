from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool
import matplotlib.pyplot as plt
import os

# Create images directory
if not os.path.exists("images"):
    os.makedirs("images")

def python_repl_plot(code: str) -> str:
    """
    Executes Python code to generate plots. 
    The code should save figures to 'images/' directory.
    Output should be the path to the saved image or text output.
    """
    repl = PythonREPL()
    
    # Prepend some setup code or ensure context
    full_code = f"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# Ensure clean state
plt.clf()
sns.set_theme(style="whitegrid")

{code}
"""
    try:
        result = repl.run(full_code)
        return f"Executed Successfully. Output: {result}"
    except Exception as e:
        return f"Error executing code: {e}"
