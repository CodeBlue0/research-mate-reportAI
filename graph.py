from langgraph.graph import StateGraph, END
from state import AgentState
from agents.planner import plan_node
from agents.researcher import research_node
from agents.writer import writer_node
from agents.visualizer import visualizer_node
from agents.critic import critic_node

def should_continue(state: AgentState):
    critique = state.get("critique")
    revision_count = state.get("revision_count")
    
    if "APPROVE" in critique.upper() or revision_count > 2:
        return "end"
    else:
        return "researcher" 
        # In a real system, we might go back to writer or researcher depending on feedback.
        # For this prototype, we'll loop back to researcher to get more data if needed, or just re-write.
        # Actually, let's go back to Writer to fix things, unless data is missing.
        # Let's simplify: Go to Writer if rejected.

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("planner", plan_node)
    workflow.add_node("researcher", research_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("visualizer", visualizer_node)
    workflow.add_node("critic", critic_node)
    
    workflow.set_entry_point("planner")
    
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "visualizer")
    workflow.add_edge("visualizer", "critic")
    
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "end": END,
            "researcher": "writer" # Looping back to writer to address feedback
        }
    )
    
    return workflow.compile()
