from typing import Any

import httpx
from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_text_splitters import RecursiveJsonSplitter

def api_retriever(urls: list[str], headers: dict[str, Any] = None) -> list[Document]:
    """
    Load and split JSON documents from the specified URLs.

    Args:
        urls (list[str]): A list of URLs to load JSON documents from.
        headers (dict[str, Any], optional): Optional headers to include in the requests.

    Returns:
        list[Document]: A list of Document objects containing the split JSON data.
    """
    # Create a JSON splitter with a maximum chunk size of 300
    splitter = RecursiveJsonSplitter(max_chunk_size=300)
    # If headers are not provided, set to an empty dictionary
    json_data = [httpx.get(url, headers=headers, timeout=10).json() for url in urls]
    # Split the JSON data into documents
    return splitter.create_documents(texts=json_data)

TOOL_NAME = "agent-retriever"
TOOL_DESCRIPTION = """
Read and return information from a JSON log file related to the configuration status of a device for the PinBike application 
to detect configuration errors for a device, focusing on the 'debug' property.
Please report only the detected issues and ignore correctly configured settings.\n
"""

def retriever_tool(thread_id: str, urls: list[str], headers: dict[str, Any] = None) -> Any:
    """
    Creates a retriever tool for the Agentic RAG model.

    Args:
        headers (dict[str, Any], optional): Optional headers to be used by the retriever. Defaults to None.

    Returns:
        Any: The created retriever tool.
        :param thread_id: (str): The unique identifier for the thread.
        :param headers: Optional headers to be used by the retriever.
        :param urls: The list of URLs to retrieve documents from.
    """
    retriever = api_retriever(urls, headers)
    return create_retriever_tool(
        retriever,
        TOOL_NAME,
        TOOL_DESCRIPTION,
    )