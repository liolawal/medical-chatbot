from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from src.prompt import *
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os


app = Flask(__name__)


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
NEW_API_KEY = os.getenv("NEW_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["NEW_API_KEY"] = NEW_API_KEY


embedding = download_embeddings()

index_name = "medical-chatbot"

vector_store = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})


llm = ChatOpenAI(model="deepseek-chat",
                  api_key=NEW_API_KEY,
                  base_url="https://api.deepseek.com/v1")


qa_prompt = PromptTemplate(
    template="""You are a professional clinical community pharmacist,
     Answer the question based on the context provided.
    If you do not know the answer say "I don't know."
    Give your answer in a simple and concise manner that is still comprehensive,robust and accurate.
    {context}

    Question: {question}
    Helpful Answer:""",
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm =llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_prompt},
    chain_type="stuff"
)


@app.route('/')
def index():
    return render_template('chat.html')


@app.route("/get",methods=["GET","POST"])
def chat():
    msg = request.form['msg']
    query = msg
    print(query)
    response = qa_chain.invoke({'query':query})
    print("Response: ",response['result'])
    return str(response['result'])


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)





