from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from utils_llm import get_llm

def writer_node(state: AgentState):
    """
    The Writer Agent:
    1. Synthesizes research data.
    2. Writes the report sections: Motivation, Theory, Deep Dive, Conclusion.
    3. Uses LaTeX for math.
    """
    plan = state.get("plan")
    research_data = state.get("research_data")
    topic = state.get("topic")
    
    print(f"--- WRITER: Writing report on {topic} ---")
    
    llm = get_llm()
    
    formatted_data = "\n".join([f"Source: {d.get('source')}\nContent: {d.get('content')[:1000]}..." for d in research_data])
    
    system_msg = """You are an elite academic writer for high school science/math reports.
    Write a comprehensive report based on the plan and research data provided.
    
    Guidelines:
    1. **Structure**: 
       - I. Introduction & Motivation (Why this topic? Connect to real life)
       - II. Theoretical Background (Textbook concepts + Academic definitions)
       - III. Deep Dive & Mathematical Analysis (Derivations, Models)
       - IV. Conclusion & Future Outlook
    2. **Tone**: Engaging, storytelling, but rigorously academic.
    3. **Math**: MUST use LaTeX format for all math functions and variables (e.g., $f(x) = x^2$, $$ \int $$).
    4. **Citations**: Cite sources inline [Source Name].
    
    Return the report in Markdown.
    """
    
    user_msg = f"""Topic: {topic}
    
    Research Plan:
    {plan}
    
    Research Data:
    {formatted_data}
    
    Write the report.
    """
    
    response = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=user_msg)])
    
    return {"draft_content": response.content}
