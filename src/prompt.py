from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA


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