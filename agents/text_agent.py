from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

def load_laptop_rows():
    try:
        db = mysql.connector.connect(
            host="localhost",
            port=3306,
            user="root",
            password="root",
            database="commerce_ai",
            connection_timeout=5,
            use_pure=True
        )

        cursor = db.cursor()
        cursor.execute("SELECT Company, Product, Ram, OS, Price_euros, CPU_model, GPU_model FROM laptops")
        rows = cursor.fetchall()
        cursor.close()
        db.close()
        return rows

    except mysql.connector.Error as err:
        print("MySQL Error:", err)
        exit(1)

def build_documents(rows):
    docs = []
    for r in rows:
        text = f"{r[0]} {r[1]} with {r[2]}GB RAM, {r[3]}, priced at €{r[4]:.2f}. CPU: {r[5]}, GPU: {r[6]}"
        docs.append(Document(page_content=text))
    return docs

def get_text_qa_chain():
    rows = load_laptop_rows()
    documents = build_documents(rows)
    print(f"Loaded {len(documents)} entries.")

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    wan_shi_tong_prompt = PromptTemplate.from_template("""
    You are Wan Shi Tong, He Who Knows a Ten Thousand Things.
    Your tone is composed, formal, and dignified. You speak with clarity.

    You are assisting users in with questions about purchases.

    NEVER fabricate information. If an answer is not in the library, respond accordingly.

    Use the following context from your library:
    {context}

    Human: {question}
    Wan Shi Tong:
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
        print(f"Wan Shi Tong: {response}")