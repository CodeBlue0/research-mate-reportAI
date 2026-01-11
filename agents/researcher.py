from langchain_core.messages import SystemMessage, HumanMessage
from tools.search_tools import get_tavily_tool, search_arxiv
from tools.rag_tool import query_textbook
from state import AgentState
from utils_llm import get_llm
import json

def research_node(state: AgentState):
    """
    The Researcher Agent:
    1. Takes the plan.
    2. Generates search queries.
    3. Executes searches (Web, ArXiv, Textbook).
    4. Aggregates findings.
    """
    plan = state.get("plan")
    print(f"--- RESEARCHER: executing research based on plan ---")
    
    llm = get_llm()
    
    # 1. Generate Queries
    query_gen_prompt = f"""Based on the following research plan, generate 3 specific search queries to gather evidence, math theories, and real-world examples.
    Return a valid JSON array of strings, e.g., ["query1", "query2", "query3"].
    
    Plan:
    {plan}
    """
    query_response = llm.invoke([HumanMessage(content=query_gen_prompt)])
    try:
        queries = json.loads(query_response.content.replace("```json", "").replace("```", "").strip())
    except:
        queries = [state.get("topic") + " advanced theory", state.get("topic") + " applications"]

    # 2. Execute Search
    tavily = get_tavily_tool()
    results = []
    
    # Textbook RAG (Already have some, but can look for specifics)
    # We add the initial context to results
    results.append({"source": "Textbook", "content": state.get("textbook_context", "")})

    for q in queries:
        print(f"Searching for: {q}")
        # Web Search
        web_res = tavily.invoke(q)
        for w in web_res:
            results.append({"source": w.get('url'), "content": w.get('content')})
        
        # ArXiv Search (Optional, if it seems academic)
        if "math" in q.lower() or "physics" in q.lower() or "theory" in q.lower():
             arxiv_res = search_arxiv(q)
             results.append({"source": "ArXiv", "content": arxiv_res})

    return {"research_data": results}
