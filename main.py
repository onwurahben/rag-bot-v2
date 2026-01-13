import gradio as gr
import logging
import uuid

# Import your PDF loader, vector store retriever, and LLM prompt logic
from pdf_loader import document_loader, split_text
from vector_store import get_retriever
from qa_chain import get_llm, prompt, build_context_and_citations
from memory_manager import wrap_chain_with_memory, save_persistent_memory

# Set up logging to debug errors
logging.basicConfig(level=logging.INFO)

# ----------------- Main QA Function -----------------
def retriever_qa(files, query, state, persistent_memory=False):
    """
    Main RAG QA function.
    Handles:
    - Multi-PDF input
    - Persistent memory
    - Context + citations
    """
    # --- Input validations ---
    if not files:
        return "Upload at least one PDF.", state, ""
    if not query.strip():
        return "Ask a question.", state, ""
    if len(query) > 5000:
        return "Question too long (max 5000 characters).", state, ""

    try:
        # --- Initialize the LLM ---
        llm = get_llm()

        # --- Load PDFs and split into chunks for the retriever ---
        all_chunks = []
        for f in files:
            docs = document_loader(f)       # Extract text from PDF
            chunks = split_text(docs)       # Split text into smaller chunks
            for c in chunks:
                c.metadata["source"] = getattr(f, "name", str(f))
            all_chunks.extend(chunks)

        # --- Create a namespace per session for the vector DB ---
        namespace = str(id(state))
        retriever = get_retriever(all_chunks, namespace)

        # --- Retrieve relevant chunks from vector DB ---
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found.", state, query

        # --- Build context + citations from retrieved chunks ---
        context_text, citations = build_context_and_citations(docs)

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
        response = chain_with_memory.invoke(
            {"question": query, "context": context_text},
            {"configurable": {"session_id": session_id}}
        ).content

        # --- Save persistent memory if enabled ---
        if persistent_memory:
            save_persistent_memory(session_id, query, response)



        # --- Combine response with citations for output ---

        # TODO: Switch to dynamic detection with a classifier LLM
        show_citations = any(word in query.lower() for word in ["what", "how", "when", "who", "cost", "price", "amount"])

        if show_citations:
           final_response = response + "\n\n📎 **Sources & Citations**\n" + citations
        else:
           final_response = response

        return final_response, state, query

    except Exception as e:
        logging.error(f"Error in QA: {e}", exc_info=True)
        return f"Error processing request: {e}", state, query





# ----------------- Modern Chatbot UI -----------------
with gr.Blocks(css="""
    .gr-chatbot {
        height: 500px;
        overflow-y: auto;
        padding: 10px;
        background-color: #F7F7F8;
        border-radius: 10px;
        border: 1px solid #E0E0E0;
    }
    .user-message {
        text-align: right;
        background-color: #DCF8C6;
        border-radius: 15px 15px 0 15px;
        padding: 10px;
        margin: 5px 0;
        display: inline-block;
        max-width: 75%;
        word-wrap: break-word;
    }
    .ai-message {
        text-align: left;
        background-color: #FFFFFF;
        border-radius: 15px 15px 15px 0;
        padding: 10px;
        margin: 5px 0;
        display: inline-block;
        max-width: 75%;
        word-wrap: break-word;
    }
""") as rag_app:

    # ----------------- Runtime State -----------------
    state = gr.State({})
    last_query = gr.State("")

    # --- App Title ---
    gr.Markdown("## 📄 Advanced Multi-PDF RAG Chatbot", elem_id="title")

    # --- Layout ---
    with gr.Row():
        with gr.Column(scale=2):
            files = gr.File(
                label="Upload PDFs",
                file_count="multiple",
                type="filepath"
            )
            persistent_checkbox = gr.Checkbox(
                label="Use persistent memory",
                value=False
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(elem_id="chatbot")
            query_input = gr.Textbox(
                placeholder="Type your question here...",
                show_label=False
            )
            ask_btn = gr.Button("Send")
            retry_btn = gr.Button("Retry Last Question")

    # ----------------- Query Handler -----------------
    def ask_wrapper(files, query_text, history, persistent):
        if not query_text.strip():
            return history, ""

        # Run RAG QA
        response, _, _ = retriever_qa(
            files,
            query_text,
            state.value,
            persistent_memory=persistent
        )

        history = history or []

        # Append messages as plain text tuples (user_msg, ai_msg)
        history.append((query_text, response))

        last_query.value = query_text
        return history, ""  # Clear input after sending

    # ----------------- Button Events -----------------
    ask_btn.click(
        ask_wrapper,
        inputs=[files, query_input, chatbot, persistent_checkbox],
        outputs=[chatbot, query_input]
    )

    retry_btn.click(
        ask_wrapper,
        inputs=[files, last_query, chatbot, persistent_checkbox],
        outputs=[chatbot, query_input]
    )

rag_app.launch()
