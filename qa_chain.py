from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import logging
import os

# ---------------- Prompt ----------------
# Load prompt from external file for better maintainability
PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompts", "qa_system_prompt.txt")

def load_prompt_template():
    if not os.path.exists(PROMPT_FILE):
        logging.error(f"Prompt file not found at {PROMPT_FILE}")
        raise FileNotFoundError(f"Missing prompt file: {PROMPT_FILE}")
    
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read()

prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=load_prompt_template()
)

# ---------------- LLM ----------------
_LLM_CACHE = None

def get_llm():
    """
    Returns a cached Groq Chat LLM instance
    """
    global _LLM_CACHE
    if _LLM_CACHE is None:
        logging.info("Initializing Groq LLM...")
        _LLM_CACHE = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0.2
        )
    return _LLM_CACHE


# ---------------- Context & Citations ---------------- 
def build_context_and_citations(docs):
    """
    Build context for the LLM and generate concise, readable footnotes for each source.
    Returns:
        context_text (str): Full context text to feed the LLM.
        citations_str (str): Clean human-friendly citations.
    """

    context_text = ""
    footnotes = []
    seen = set()  # Prevent duplicate references

    for d in docs:

        # ---- Extract metadata safely ----
        source_path = d.metadata.get("source", d.metadata.get("file_path", "document"))
        filename = source_path.split("\\")[-1].split("/")[-1]

        # Handle different page metadata keys safely
        raw_page = (
            d.metadata.get("page")
            or d.metadata.get("page_number")
            or d.metadata.get("page_index")
        )

        # ---- Normalize page label safely ----
        if isinstance(raw_page, int):
            # Most PDF loaders use zero-indexed pages → convert to human readable
            page_label = f"Page {raw_page + 1 if raw_page >= 0 else raw_page}"
        elif isinstance(raw_page, str) and raw_page.strip():
            page_label = f"Page {raw_page.strip()}"
        else:
            # If page truly unknown, just don't force anything fake
            page_label = ""

        # ---- Deduplication key ----
        key = (filename, page_label)
        if key in seen:
            continue
        seen.add(key)

        # ---- Extract excerpt for context ----
        excerpt = d.page_content.replace("\n", " ").strip()
        context_text += excerpt + "\n\n"

        # ---- Build short readable summary line ----
        parts = [s.strip() for s in excerpt.split(".") if s.strip()]

        if len(parts) == 0:
            summary = "Referenced content"
        elif len(parts[0]) < 20 and len(parts) > 1:
            summary = (parts[0] + ". " + parts[1]).strip()
        else:
            summary = parts[0]

        # ---- Final modern citation format ----
        # Example:
        # [1] Marketing.pdf — Page 2: Cost breakdown for AI workflow tools.
        label = f" — {page_label}" if page_label else ""
        footnotes.append(
            f"[{len(footnotes)+1}] {filename}{label}: {summary}."
        )

    # ---- Join nicely formatted citations ----
    citations_str = "\n".join(footnotes) if footnotes else ""
    return context_text, citations_str
