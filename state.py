from typing import TypedDict, List, Annotated
import operator

class AgentState(TypedDict):
    topic: str
    plan: str
    research_data: Annotated[List[dict], operator.add]
    textbook_context: str
    draft_content: str
    critique: str
    revision_count: int
    final_report: str
    image_paths: List[str]
