from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

def document_loader(file):
    path = file.name if hasattr(file, 'name') else file
    logging.info(f"Loading PDF from: {path}")
    try:
        loader = PyMuPDFLoader(path)
        docs = loader.load()
        logging.info(f"Loaded {len(docs)} pages.")
        return docs
    except Exception as e:
        logging.error(f"PDF parsing error for {path}: {e}", exc_info=True)
        raise ValueError(f"PDF could not be parsed: {e}")

def split_text(docs, chunk_size=800, chunk_overlap=300):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_documents(docs)
