---
title: "RAG PDF Chatbot"
emoji: "📄"
colorFrom: "blue"
colorTo: "green"
sdk: gradio
sdk_version: "5.50.0"
app_file: app.py
pinned: true
---


# 📄 DocQuery - RAG Document Assistant (Groq + LCEL + Pinecone)

Upload one or more PDF documents and ask questions. Fully cloud-based with **Pinecone vector DB** 🧠 and powered by Llama 3 via **Groq API** with LCEL-style prompting.

## Features

- Retrieve answers from multiple PDF documents using RAG (Retrieval-Augmented Generation)  
- Top-K chunk retrieval ensures relevant context is used  
- Fast, actionable, and professional AI responses  
- Embeddings powered by **sentence-transformers/all-MiniLM-L6-v2**  
- Supports multiple PDF document uploads and queries  
- Persistent memory across sessions 
- chat interface with retry functionality


## Usage

1. Upload PDF files.  
2. Enter your question in the textbox.  
3. Click submit to get answers.  
4. Click retry to get answers for the last question.  

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
