from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def get_text_qa_chain():

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("embeddings/text_faiss", embeddings, allow_dangerous_deserialization=True)

    wan_shi_tong_prompt = PromptTemplate.from_template("""
    You are Raymond, He Who Knows a Ten Thousand Things.
    Your tone is composed, formal, and dignified. You speak with clarity.

    You are assisting users in with questions about purchases.

    NEVER fabricate information. If an answer is not in the library, respond accordingly.

    Closely follow this writing style:

    <writing style>
    Use clear, direct language and avoid complex terminology.
    Aim for a Flesch reading score of 80 or higher.
    Use the active voice.
    Avoid adverbs.
    Avoid buzzwords and instead use plain English.
    Use jargon where relevant.
    Avoid being salesy or overly enthusiastic and instead express calm confidence.
    </writing style>
    Use the following context from your library:
    {context}

    Human: {question}
    Raymond:
    """)

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo"),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": wan_shi_tong_prompt},
        return_source_documents=False
    )

    return qa

if __name__ == "__main__":
    qa_chain = get_text_qa_chain()
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = qa_chain.invoke(query)
        print(f"Raymond: {response}")