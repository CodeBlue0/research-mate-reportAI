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
    
    system_msg = """You are an expert academic planner for high school/undergraduate students.
    Your goal is to create a structured research plan based on a given topic and relevant textbook content.
    
    1. Analyze the topic.
    2. Connect it to specific curriculum units found in the textbook context.
    3. Suggest a "Deep Dive" angle (e.g., Mathematical optimization, Real-world application).
    4. Outline 3-4 key research questions.
    
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
