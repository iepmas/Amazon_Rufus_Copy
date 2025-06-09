import os
import pickle
import numpy as np
import faiss
import torch
from PIL import Image
import clip
from typing import List, Tuple
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def load_faiss_index(index_path):
    return faiss.read_index(str(index_path))

def load_metadata(metadata_path):
    with open(metadata_path, "rb") as f:
        return pickle.load(f)

def load_clip_model():
    model, preprocess = clip.load("ViT-B/32")
    return model, preprocess

def encode_image(image_path, model, preprocess):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_image(image).squeeze().numpy()
        embedding = embedding / np.linalg.norm(embedding)  # normalize
    return embedding.astype("float32")

def search_similar_images(
    image_path,
    index,
    metadata,
    model,
    preprocess,
    k = 5
):
    query_vector = encode_image(image_path, model, preprocess).reshape(1, -1)
    scores, indices = index.search(query_vector, k)
    return [(float(scores[0][i]), metadata[indices[0][i]]) for i in range(k)]

def init_image_search():
    index = load_faiss_index("embeddings/fashion_faiss.index")
    metadata = load_metadata("embeddings/fashion_metadata.pkl")
    model, preprocess = load_clip_model()
    return index, metadata, model, preprocess

# This is just for testing
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    INDEX_PATH = BASE_DIR / "embeddings/fashion_faiss.index"
    METADATA_PATH = BASE_DIR / "embeddings/fashion_metadata.pkl"
    QUERY_IMAGE = "data/images/10054.jpg"  # Todo: Update to receive input from front-end

    index = load_faiss_index(INDEX_PATH)
    metadata = load_metadata(METADATA_PATH)
    model, preprocess = load_clip_model()

    results = search_similar_images(QUERY_IMAGE, index, metadata, model, preprocess)

    print(f"Top matches for {Path(QUERY_IMAGE).name}:\n")
    for score, data in results:
        print(f"Score: {score:.4f} | ID: {data['id']} | Category: {data['subCategory']} | Name: {data['productDisplayName']}")
