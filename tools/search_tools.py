from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper

def get_tavily_tool():
    """Returns a Tavily search tool configured for high quality results."""
    return TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        include_images=False, # We might handle images separately
        # include_domains=[".edu", ".gov", ".org"], # Optional filter
    )

def search_arxiv(query: str) -> str:
    """Useful for searching academic papers on ArXiv."""
    arxiv = ArxivAPIWrapper(
        top_k_results=3,
        ARXIV_MAX_QUERY_LENGTH=300,
        load_max_docs=3,
        load_all_available_meta=False,
        doc_content_chars_max=1000  # truncate specifically for RAG
    )
    return arxiv.run(query)
