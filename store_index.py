import os
from dotenv import load_dotenv
from src.helper import extract_documents, filter_to_minimal_doc, create_chunk, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
NEW_API_KEY = os.getenv("NEW_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["NEW_API_KEY"] = NEW_API_KEY

extracted_documents = extract_documents("Data")
filtered_docs = filter_to_minimal_doc(extracted_documents)
text_chunks = create_chunk(filtered_docs)
embedding = download_embeddings()

pinecone_api_key = PINECONE_API_KEY

pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

#insert documents into the vectorstore
vectorstore = PineconeVectorStore.from_documents(documents=text_chunks,
                                  embedding=embedding,
                                  index_name=index_name)

