import pandas as pd
import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = "data/styles.csv"
INDEX_SAVE_DIR = "embeddings/text_faiss"

def load_rows_from_csv(limit=5000):
    df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
    df = df.dropna(subset=[
        "gender", "masterCategory", "subCategory", "articleType",
        "baseColour", "season", "year", "usage", "productDisplayName"
    ])
    df = df.head(limit)
    rows = df[[
        "gender", "masterCategory", "subCategory", "articleType",
        "baseColour", "season", "year", "usage", "productDisplayName"
    ]].values.tolist()
    return rows

def build_documents(rows):
    docs = []
    for r in rows:
        text = (
            f"{r[0]} {r[1]} > {r[2]} > {r[3]} - {r[8]} in {r[4]} color "
            f"for {r[6]} {r[5]} ({r[7]} use)"
        )
        docs.append(Document(page_content=text))
    return docs

def main():
    rows = load_rows_from_csv(limit=44000)
    documents = build_documents(rows)
    print(f"Loaded {len(documents)} entries.")

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    print(f"Saving FAISS index to '{INDEX_SAVE_DIR}'...")
    vectorstore.save_local(INDEX_SAVE_DIR)
    print("Done.")

if __name__ == "__main__":
    main()