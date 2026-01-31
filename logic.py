import logging
import gradio as gr
import uuid
import time
import os

# Configure logging with proper timestamps and formatting
LOG_FILE = "rag_bot.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RAG-Logic")

# Import your PDF loader, vector store retriever, and LLM prompt logic
from pdf_loader import document_loader, split_text
from vector_store import get_retriever, is_namespace_populated, RETRIEVER_CACHE
from qa_chain import get_llm, prompt, build_context_and_citations
from memory_manager import wrap_chain_with_memory, save_persistent_memory


# ----------------- Main QA Function -----------------
def retriever_qa(files, query, state, persistent_memory=False):
    """
    Main RAG QA function with performance optimizations.
    Handles:
    - Multi-PDF input
    - Persistent memory
    - Context + citations
    """
    start_total = time.time()
    
    # --- Input validations ---
    if not files:
        logger.warning("Validation failed: No files uploaded.")
        gr.Warning("Upload at least one PDF.")
        return None, state, ""
    if not query.strip():
        logger.warning("Validation failed: Empty query.")
        gr.Warning("Ask a question.")
        return None, state, ""
    if len(query) > 5000:
        logger.warning(f"Validation failed: Query too long ({len(query)} chars).")
        gr.Warning("Question too long (max 5000 characters).")
        return None, state, ""

    try:
        # --- Prepare namespace based on file content ---
        # Generate a unique namespace hash based on the file names/paths.
        # This ensures that if the file set changes, we get a new namespace
        # and trigger a new indexing process.
        if not files:
             # Should be caught by validation above, but safe fallback
             return None, state, ""
             
        # Create a stable identifier for the set of files
        import hashlib
        file_identifiers = sorted([getattr(f, "name", str(f)) for f in files])
        files_string = "".join(file_identifiers)
        namespace = hashlib.md5(files_string.encode()).hexdigest()
        logger.info(f"Generated namespace hash: {namespace} for files: {file_identifiers}")
        
        # --- Setup/Retriever Optimization ---
        start_setup = time.time()
        retriever = None
        
        # Check cache first
        if namespace in RETRIEVER_CACHE:
            retriever = RETRIEVER_CACHE[namespace]
            logger.info("Using cached retriever.")
        else:
            # Check if namespace is already populated in Pinecone
            if is_namespace_populated(namespace):
                logger.info(f"Namespace {namespace} already populated. Skipping PDF processing.")
                retriever = get_retriever([], namespace)
            else:
                logger.info(f"Namespace {namespace} empty. Processing PDFs...")
                all_chunks = []
                for f in files:
                    docs = document_loader(f)       # Extract text from PDF
                    chunks = split_text(docs)       # Split text into smaller chunks
                    for c in chunks:
                        c.metadata["source"] = getattr(f, "name", str(f))
                    all_chunks.extend(chunks)
                retriever = get_retriever(all_chunks, namespace)
        
        setup_time = time.time() - start_setup

        # --- Retrieve relevant chunks from vector DB ---
        start_search = time.time()
        docs = retriever.invoke(query)
        search_time = time.time() - start_search
        
        if not docs:
            return "No relevant information found.", state, query

        # --- Build context + citations from retrieved chunks ---
        context_text, citations = build_context_and_citations(docs)

        # --- LLM Optimization & Inference ---
        start_llm = time.time()
        llm = get_llm()
        
        # --- Prepare session ID for memory management ---
        if "session_id" not in state:
            state["session_id"] = f"session-{uuid.uuid4()}"
        session_id = state["session_id"]

        # --- Wrap the chain with memory support ---
        chain = prompt | llm
        chain_with_memory, _ = wrap_chain_with_memory(
            chain,
            session_id=session_id,
            persistent=persistent_memory
        )

        # --- Run the chain, passing session_id for memory tracking ---
        response_obj = chain_with_memory.invoke(
            {"question": query, "context": context_text},
            {"configurable": {"session_id": session_id}}
        )

        response = response_obj.content
        llm_time = time.time() - start_llm

        # --- Logging Performance Metrics ---
        total_time = time.time() - start_total
        logger.info(f"PERF | Setup: {setup_time:.2f}s | Search: {search_time:.2f}s | LLM: {llm_time:.2f}s | Total: {total_time:.2f}s")

        # --- Save persistent memory if enabled ---
        if persistent_memory:
            save_persistent_memory(session_id, query, response)

        # --- Combine response with citations for output ---

        # TODO: Switch to dynamic detection with a classifier LLM
        show_citations = any(word in query.lower() for word in [
        "policy", "refund", "price", "cost", "services",
        "terms", "scope", "contract", "what does", "list"
    ])
      
        if show_citations:
           final_response = response + "\n\n📎 **Sources & Citations**\n" + citations
        else:
           final_response = response

        return final_response, state, query

    except Exception as e:
        logger.error(f"Error in QA: {e}", exc_info=True)
        return f"Error processing request: {e}", state, query
