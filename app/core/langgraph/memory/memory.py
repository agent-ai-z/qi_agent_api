import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain.embeddings import OpenAIEmbeddings

from core import logger

embedchain = OpenAIEmbeddings(model="text-embedding-3-small")

def qdrant_connection():
    """
    Establishes a connection to the Qdrant client using environment variables.

    Raises:
        Exception: If the QDRANT_URL or QDRAND_PORT environment variables are not set.

    Returns:
        QdrantClient: An instance of the Qdrant client.
    """
    # Check if the QDRANT_URL environment variable is set
    url = os.getenv("QDRANT_URL")
    if url is None or url == "":
        raise Exception("The QDRANT_URL environment variable must be provided")

    # Qdrant client
    return QdrantClient(url=url)

# Establish a connection to Qdrant
qdrant_client = qdrant_connection()

def empty_collection(collection: str) -> bool:
    """
    Empty the specified collection.

    Args:
        collection (str, optional): The name of the collection. Defaults to COLLECTION_NAME.
    """
    is_deleted = qdrant_client.delete_collection(collection_name=collection)
    logger.info(f"Collection '{collection}' emptied successfully!")
    return is_deleted

def vector_store(collection: str) -> QdrantVectorStore:
    """
    Get or create a Qdrant vector store for the specified collection.

    Args:
        collection (str, optional): The name of the collection. Defaults to COLLECTION_NAME.

    Returns:
        QdrantVectorStore: The Qdrant vector store for the specified collection.
    """
    collections_list = qdrant_client.get_collections()
    existing_collections = [col.name for col in collections_list.collections]

    # Check if the collection exists, if not, create it
    if collection not in existing_collections:
        qdrant_client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        logger.info(f"Collection '{collection}' creata con successo!")
    else:
        logger.info(f"La collection '{collection}' esiste già.")

    # Initialize Qdrant vector store from the existing collection
    return QdrantVectorStore.from_existing_collection(
        embedding=embedchain,
        collection_name=collection,
        url=os.getenv("QDRANT_URL")
    )