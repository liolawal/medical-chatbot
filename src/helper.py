from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec
from typing import List

# Function to extract documents from a directory
def extract_documents(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


# Function to filter documents to minimal representation(just page content and page for the metadata)
def filter_to_minimal_doc(docs:List[Document]) -> List[Document]:
    minimal_docs = []

    for doc in docs:
        page = doc.metadata.get('page','')
        page_content = doc.page_content
        new_doc = Document(page_content=page_content, metadata={'page': page})
        minimal_docs.append(new_doc)
    return minimal_docs


# Function to create chunks from the filtered documents
def create_chunk(filtered_docs):
    split = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
    chunks = split.split_documents(filtered_docs)
    return chunks


#download huggingface embeddings
def download_embeddings():
    """Download and return the Hugging Face embeddings model."""
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


