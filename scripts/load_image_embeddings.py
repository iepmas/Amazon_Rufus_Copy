import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import faiss
import pickle
from PIL import Image
from torchvision import models, transforms
import torch
from tqdm import tqdm

# Load ResNet18
resnet = models.resnet18(pretrained=True)
resnet.eval()
model = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove final classifier

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load data
csv_path = "data/styles.csv"
image_folder = "data/images/"
max_images = 2000

df = pd.read_csv(csv_path, on_bad_lines="skip").head(max_images)
embeddings = []
metadata = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_id = str(row["id"])
    fname = f"{img_id}.jpg"
    img_path = os.path.join(image_folder, fname)

    try:
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            emb = model(input_tensor).squeeze().numpy()
            emb = emb.flatten()  # ResNet outputs (512, 1, 1)
            emb = emb / np.linalg.norm(emb)  # normalize
            embeddings.append(emb)

        metadata.append({
            "id": img_id,
            "filename": fname,
            "productDisplayName": row["productDisplayName"],
            "subCategory": row["subCategory"],
            "gender": row["gender"]
        })
        
    except Exception as e:
        print(f"Failed on {fname}: {e}")

# Save to FAISS
embedding_matrix = np.vstack(embeddings).astype("float32")
index = faiss.IndexFlatIP(embedding_matrix.shape[1])
index.add(embedding_matrix)

faiss.write_index(index, "./embeddings/resnet_faiss.index")
with open("./embeddings/resnet_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print(f"Stored {len(metadata)} ResNet18 embeddings.")
