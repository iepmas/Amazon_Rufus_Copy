import os
import pandas as pd
from PIL import Image
import torch
import clip
import numpy as np
import faiss
import pickle
from tqdm import tqdm

model, preprocess = clip.load("ViT-B/32")

csv_path = "data/styles.csv"
image_folder = "data/images/"
max_images = 44000

df = pd.read_csv(csv_path, on_bad_lines="skip")
df = df.head(max_images)  # Too many files lol

embeddings = []
metadata = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_id = str(row["id"])
    fname = f"{img_id}.jpg"
    img_path = os.path.join(image_folder, fname)

    try:
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            emb = model.encode_image(image).squeeze().numpy()
            emb = emb / np.linalg.norm(emb)  # normalize
            embeddings.append(emb)

            metadata.append({
                "id": img_id,
                "filename": fname
            })

    except Exception as e:
        print(f"Failed on {fname}: {e}")

# Convert list to numpy to store in FAISS
embedding_matrix = np.vstack(embeddings).astype("float32")
index = faiss.IndexFlatIP(embedding_matrix.shape[1])
index.add(embedding_matrix)

# Save index and metadata
faiss.write_index(index, ".embeddings/fashion_faiss.index")
with open(".embeddings/fashion_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print(f"Stored {len(metadata)} embeddings and metadata.")