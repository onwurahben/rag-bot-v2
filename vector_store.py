from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os, logging
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "rag-pdf-bot"
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in [index.name for index in pc.list_indexes()]:
    pc.create_index(name=INDEX_NAME, dimension=384, metric="cosine", spec={"serverless": {"cloud":"aws"}})
pc_index = pc.Index(INDEX_NAME)

_embedding_model_cache = None
def get_embedding_model():
    global _embedding_model_cache
    if _embedding_model_cache is None:
        logging.info("Initializing embedding model...")
        _embedding_model_cache = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model_cache

RETRIEVER_CACHE = {}

def get_retriever(chunks, namespace, k=5):
    """
    Returns a retriever for a given namespace.
    Caches the retriever in memory to avoid redundant connections.
    """
    global RETRIEVER_CACHE
    
    if namespace in RETRIEVER_CACHE:
        logging.info(f"Returning cached retriever for namespace: {namespace}")
        return RETRIEVER_CACHE[namespace]

    logging.info(f"Initializing new retriever for namespace: {namespace}")
    vectordb = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        index_name=INDEX_NAME,
        namespace=namespace
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    RETRIEVER_CACHE[namespace] = retriever
    return retriever

def is_namespace_populated(namespace):
    """
    Checks if a namespace exists and has vectors in Pinecone.
    """
    try:
        stats = pc_index.describe_index_stats()
        return namespace in stats.get("namespaces", {})
    except Exception as e:
        logging.error(f"Error checking namespace stats: {e}")
        return False
