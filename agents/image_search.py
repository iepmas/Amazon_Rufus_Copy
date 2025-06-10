import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pickle
import numpy as np
import faiss
import torch
from PIL import Image
from torchvision import models, transforms
from pathlib import Path
from typing import List, Tuple
from image_model import get_resnet_model, get_preprocess

# Load ResNet18 model
resnet = get_resnet_model()
resnet.eval()
model = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove classifier layer

# Define image transform
transform = get_preprocess()

def load_faiss_index(index_path):
    return faiss.read_index(str(index_path))

def load_metadata(metadata_path):
    with open(metadata_path, "rb") as f:
        return pickle.load(f)

def encode_image(image_path: str):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        emb = model(input_tensor).squeeze().detach().cpu().numpy()
        emb = emb / np.linalg.norm(emb)  # normalize
    return emb.astype("float32")

def search_similar_images(
    image_path: str,
    index,
    metadata,
    k: int = 5
):
    query_vector = encode_image(image_path).reshape(1, -1)
    scores, indices = index.search(query_vector, k)
    return [(float(scores[0][i]), metadata[indices[0][i]]) for i in range(k)]

def init_image_search():
    index = load_faiss_index("embeddings/resnet_faiss.index")
    metadata = load_metadata("embeddings/resnet_metadata.pkl")
    return index, metadata

# Testing
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    INDEX_PATH = BASE_DIR / "embeddings/resnet_faiss.index"
    METADATA_PATH = BASE_DIR / "embeddings/resnet_metadata.pkl"
    QUERY_IMAGE = "data/images/10054.jpg" # Todo: Update to receive input from front-end

    index = load_faiss_index(INDEX_PATH)
    metadata = load_metadata(METADATA_PATH)

    results = search_similar_images(QUERY_IMAGE, index, metadata)

    print(f"Top matches for {Path(QUERY_IMAGE).name}:\n")
    for score, data in results:
        print(f"Score: {score:.4f} | ID: {data['id']} | Filename: {data['filename']}")
