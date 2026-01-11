from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from utils_llm import get_llm

def critic_node(state: AgentState):
    """
    The Critic Agent:
    1. Reviews the Draft and Visuals.
    2. Checks for LaTeX correctness, Reference definitions, and Logic.
    3. Decides whether to Approve or Reject.
    """
    final_report = state.get("final_report")
    revision_count = state.get("revision_count") or 0
    
    print(f"--- CRITIC: Reviewing report (Revision {revision_count}) ---")
    
    if revision_count >= 2:
        # Stop infinite loops
        return {"critique": "Maximum revisions reached. Formatting approved.", "revision_count": revision_count + 1}
    
    llm = get_llm()
    
    system_msg = """You are a strict high school teacher reviewing a student's research report.
    Check for:
    1. **Mathematical Accuracy**: Area equations correct? LaTeX formatting used?
    2. **Citations**: Are sources cited?
    3. **Logical Flow**: Does it make sense?
    
    If excellent, reply with "APPROVE".
    If needs improvement, provide specific feedback.
    """
    
    user_msg = f"""Report Content:
    {final_report[:5000]}...
    
    Review this report.
    """
    
    response = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=user_msg)])
    
    critique = response.content
    
    if "APPROVE" in critique.upper():
        print("--- CRITIC: Report Approved ---")
    else:
        print("--- CRITIC: Report Rejected, feedback provided ---")
        
    return {
        "critique": critique,
        "revision_count": revision_count + 1
    }
