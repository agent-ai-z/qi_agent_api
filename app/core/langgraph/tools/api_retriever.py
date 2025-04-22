from typing import Any

import httpx
from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_text_splitters import RecursiveJsonSplitter
from core import logger
from ..memory.memory import vector_store, empty_collection  # Import vector_store from the appropriate module
import asyncio

TOOL_NAME = "agent-retriever"
TOOL_DESCRIPTION = """
Read and return information from a JSON log file related to the configuration status of a device for the PinBike application 
to detect configuration errors for a device, focusing on the 'debug' property.
Please report only the detected issues and ignore correctly configured settings.\n
"""

def api_load(urls: list[str], headers: dict[str, Any] = None) -> list[Document]:
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

async def process_documents(qdrant_vs, doc_splits, search: bool = False):
    """
    Processes documents by first deleting existing documents with matching content
    and then adding the new document splits to the Qdrant vector store.

    Args:
        qdrant_vs: The Qdrant vector store instance.
        doc_splits: The list of document splits to process.
        :param qdrant_vs:
        :param doc_splits:
        :param search:
    """
    if search:
        # Perform a similarity search to get the document IDs
        results = await qdrant_vs.asimilarity_search(doc_splits[0].page_content)
        logger.info(f"Similarity search results: {results}")
        if results:
            # Get all ids from the documents
            doc_ids = [doc.metadata.get("id") for doc in results]
            logger.info(f"Processing document: {doc_ids}")
            # Filter doc_ids to remove all None values
            doc_ids = list(filter(None, doc_ids))
            if len(doc_ids) > 0:
                await qdrant_vs.adelete(ids=doc_ids)
                logger.info(f"Collection ids '{doc_ids}' deleted successfully!")

    await qdrant_vs.aadd_documents(doc_splits)
    logger.info(f"{len(doc_splits)} Documents added successfully!")

def get_vs_retriever(thread_id: str, urls: list[str], headers: dict[str, Any] = None) -> Any:
    """
    Get the Qdrant vector store retriever for the specified collection.

    Args:
        thread_id (str): The unique identifier for the thread.
        urls (list[str]): A list of URLs to load JSON documents from.
        headers (dict[str, Any], optional): Optional headers to include in the requests.

    Returns:
        Any: The Qdrant retriever for the specified collection.
    """
    try:
        # Create a collection name
        collection = f"agent_api_{thread_id}_{os.getenv('APP_ENV', 'development')}"

        # Check if the collection is empty
        is_empty = empty_collection(collection)
        if is_empty:
            logger.info(f"Collection '{collection}' emptied successfully!")

        # Create a Qdrant vector store
        qdrant_vs = vector_store(collection=collection)

        # Load and split the documents
        doc_splits = api_load(urls, headers)

        # Process the documents asynchronously
        asyncio.create_task(process_documents(qdrant_vs, doc_splits, not is_empty))

        # Return the Qdrant retriever
        return qdrant_vs.as_retriever()
    except Exception as e:
        logger.error(f"Error in get_vs_retrieve: {str(e)}")
        raise e
    
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
    retriever = get_vs_retriever(thread_id, urls, headers)
    return create_retriever_tool(
        retriever,
        TOOL_NAME,
        TOOL_DESCRIPTION,
    )