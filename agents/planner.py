from langchain_core.messages import SystemMessage, HumanMessage
from tools.rag_tool import query_textbook
from state import AgentState
from utils_llm import get_llm

def plan_node(state: AgentState):
    """
    The Planner Agent:
    1. Analyzes the user's topic.
    2. Checks textbooks for curriculum mapping.
    3. Expands the topic into a research plan.
    """
    topic = state.get("topic")
    print(f"--- PLANNER: Processing topic '{topic}' ---")

    # 1. RAG Check
    textbook_context = query_textbook(topic)
    
    # 2. Planning Prompt
    llm = get_llm()
    
    system_msg = """You are an academic planner for a high school advanced exploration project.
    Your goal is to stick to the "Student-Researcher" persona.
    
    1. Analyze the topic to identify:
       - The Base Concept (from High School Curriculum).
       - The Advanced Target Concept (Career usage).
    2. Create a research plan that investigates:
       - How the Base Concept differs in real-world application.
       - The mathematical/theoretical gap between the two.
    3. Outline 3-4 key questions that a curious student would ask.
    
    Output the plan in Markdown.
    """
    
    user_msg = f"""Topic: {topic}
    
    Textbook Context:
    {textbook_context}
    
    Create a research plan.
    """
    
    response = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=user_msg)])
    
    return {
        "plan": response.content,
        "textbook_context": textbook_context
    }
