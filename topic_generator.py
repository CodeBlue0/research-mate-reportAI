from langchain_core.prompts import ChatPromptTemplate
from utils_llm import get_llm

def generate_topic(career_path: str, curriculum: str) -> str:
    """
    Generates a research topic based on the user's career path and desired curriculum.
    """
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a mentor guiding high school students in their advanced subject exploration (심화탐구). Your goal is to suggest topics that bridge a core high school concept with a specific advanced application."),
        ("user", "My career goal is to become a {career_path}. In school, I am currently learning {curriculum}. Please suggest a refined research topic. Requirements:\n1. Select ONE core concept from the high school {curriculum}.\n2. Connect it to ONE specific, advanced concept relevant to {career_path}.\n3. The topic should sound like a student's exploration journey (e.g., 'Exploring [Advanced Concept] through [High School Concept]').\n4. Output ONLY the topic title in Korean.")
    ])
    
    chain = prompt | llm
    
    response = chain.invoke({
        "career_path": career_path,
        "curriculum": curriculum
    })
    
    return response.content.strip()
