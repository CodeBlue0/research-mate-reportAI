from langchain_core.messages import SystemMessage, HumanMessage
from tools.plot_tools import python_repl_plot
from state import AgentState
from utils_llm import get_llm
import re

def visualizer_node(state: AgentState):
    """
    The Visualizer Agent:
    1. Analyzes the draft content to find opportunities for visualization.
    2. Generates Python code to create plots.
    3. Executes the code.
    """
    draft = state.get("draft_content")
    print(f"--- VISUALIZER: Generating plots ---")
    
    llm = get_llm()
    
    prompt = f"""Based on the following report draft, identify 1 key mathematical concept or data relationship that can be visualized with a Python (Matplotlib/Seaborn) graph.
    
    Draft Snippet:
    {draft[:3000]}
    
    Generate the Python code to draw this graph.
    - Use `plt.savefig('images/plot.png')` to save it.
    - Do not use `plt.show()`.
    - Handle empty data scenarios with dummy data that represents the concept if real data isn't explicitly provided, but prefer real functions (e.g., plotting a Catenary curve if mentioned).
    
    Return ONLY the python code.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    code = response.content.replace("```python", "").replace("```", "").strip()
    
    result = python_repl_plot(code)
    
    image_path = "images/plot.png" # Assuming fixed name for prototype, or parse from code
    
    # Append image to the final report if successful
    final_report = draft + f"\n\n![Generated Plot]({image_path})\n\n"
    
    return {
        "image_paths": [image_path],
        "final_report": final_report
    }
