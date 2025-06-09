release: python scripts/build_text_faiss.py && python scripts/load_image_embeddings.py
web: python -m uvicorn api.main:app --host=0.0.0.0 --port=${PORT}
