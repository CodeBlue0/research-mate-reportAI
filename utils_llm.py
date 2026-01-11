import os
from langchain_openai import ChatOpenAI

def get_llm():
    """
    Returns a configured ChatOpenAI instance.
    Supports DeepSeek via OPENAI_API_BASE and LLM_MODEL env vars.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    model_name = os.getenv("LLM_MODEL", "gpt-4o")

    # If using DeepSeek, base_url should be https://api.deepseek.com
    # and model might be deepseek-chat
    
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0.7
    )
