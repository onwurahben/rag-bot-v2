---
app_file: app.py
title: "RAG PDF Chatbot"
emoji: "📄"
colorFrom: "blue"
colorTo: "green"
sdk: gradio
pinned: true
---


# 📄 RAG PDF Chatbot (Groq + LCEL + Pinecone)

Upload a PDF and ask questions. Fully cloud-based with **Pinecone vector DB** 🧠 and powered by **Groq LLM** with LCEL-style prompting.

## Features

- Retrieve answers from any PDF using RAG (Retrieval-Augmented Generation)  
- Top-K chunk retrieval ensures relevant context is used  
- Fast, actionable, and professional AI responses  
- Embeddings powered by **sentence-transformers/all-MiniLM-L6-v2**  
- Supports multiple PDF uploads and queries  

## Usage

1. Upload a PDF file.  
2. Enter your question in the textbox.  
3. Click submit to get answers.  

## Notes

- The `.env` file containing your **Groq API Key** and **Pinecone API Key** **must not** be committed to the repository.  
- Ensure all dependencies in `requirements.txt` are installed.  

## Dependencies

- gradio  
- langchain-community  
- pinecone-client  
- langchain-groq  
- sentence-transformers  
- python-dotenv  

---

Powered by **onwurahben**  
