"""
Build the FAISS index for Wikidata ICL examples.

Usage:
    python build_wikidata_faiss.py

Expects:  data/datasets/qald_9_plus_train_wikidata_en.json
Produces: data/experience-pool/qald_9_plus_train_wikidata_en/
"""

import json
import os

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
JSON_PATH = "./data/datasets/qald_9_plus_train_wikidata_en.json"
OUTPUT_DIR = "./data/experience-pool/qald_9_plus_train_wikidata_en"


def main():
    print(f"Loading questions from {JSON_PATH} ...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for idx, item in enumerate(data):
        question = item["question"]
        doc = Document(page_content=question, metadata={"seq_num": idx + 1})
        docs.append(doc)

    print(f"Loaded {len(docs)} questions. Building embeddings with {EMBEDDING_MODEL} ...")
    model_kwargs = {"device": "cpu", "model_kwargs": {"use_safetensors": False}}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    db = FAISS.from_documents(docs, embeddings)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    db.save_local(OUTPUT_DIR)
    print(f"FAISS index saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
