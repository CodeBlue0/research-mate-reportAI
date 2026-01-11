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
    
    system_msg = """You are a high school student writing a top-tier "Advanced Subject Exploration Report" (심화탐구 보고서).
    Write the report based on the plan and research data provided.
    
    Guidelines:
    1. **Structure AND Content**:
       - **I. Introduction & Motivation**: Explicitly state: "I became interested in this topic while studying [Subject] in class..." Explain the specific curiosity that led to this research.
       - **II. Theoretical Background**: Explain the High School concept clearly first, then introduce the Advanced concept. Show the bridge between them.
       - **III. Deep Dive & Analysis**: The core content. Explain the advanced concept using the high school knowledge as a base. Use LaTeX for math.
       - **IV. Conclusion & Realization**: Conclude with: "Through this exploration, I realized that [High School Concept] is applied in [Advanced Field] by..."
    
    2. **Tone**: Academic but personal. It's a record of *your* learning process.
    3. **Formatting**: DO NOT use markdown bolding (**text**) or italics (*text*) in the body paragraphs. Keep the text clean and plain. Use headers (#, ##) for sections only.
    4. **Math**: MUST use LaTeX format for all math functions and variables (e.g., $f(x) = x^2$).
    5. **Language**: Korean (Hangul).
    
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
