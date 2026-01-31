import gradio as gr
import logging

# Import core RAG logic from logic.py
from logic import retriever_qa

# Set up logging for UI-level info (inherited from logic.py)
logger = logging.getLogger("RAG-App")

# Load CSS content to force bypass Gradio/Browser caching
with open("style_v2.css", "r", encoding="utf-8") as f:
    css_content = f.read()

# ----------------- GRADIO Chatbot UI -----------------
with gr.Blocks(css=css_content, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as rag_app:

    # ----------------- Runtime State -----------------
    state = gr.State({})
    last_query = gr.State("")

    # --- App Title ---
    gr.HTML('<h1 id="title" style="text-align: center; width: 100%; margin: 40px 0;">📄 DocQuery: RAG Document Assistant</h1>')

    # --- Main Layout ---
    with gr.Row():
        with gr.Column(scale=1, elem_classes="pdf-section"):
            gr.Markdown("### 📥 Source Documents")
            files = gr.File(
                label="Upload PDFs",
                file_count="multiple",
                type="filepath"
            )
            persistent_checkbox = gr.Checkbox(
                label="Keep memory across sessions",
                value=False
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(elem_id="chatbot", show_label=False)
            
            # --- Unified Input Group (Floating Button Layout) ---
            with gr.Group(elem_id="input-group"):
                query_input = gr.Textbox(
                    placeholder="Ask anything about the PDFs...",
                    show_label=False,
                    lines=4,
                    max_lines=10,
                    container=False # Removes internal Gradio box
                )
                ask_btn = gr.Button("➤", elem_classes="send-btn")
            
            # --- Secondary Action Row ---
            retry_btn = gr.Button("🔄 Retry Last Query", elem_classes="retry-btn")

    # ----------------- Query Handler -----------------
    def ask_wrapper(files, query_text, history, persistent):
        # Run RAG QA
        response, _, _ = retriever_qa(
            files,
            query_text,
            state.value,
            persistent_memory=persistent
        )

        if response is None:
            return history, query_text # Warning already shown in logic.py

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
