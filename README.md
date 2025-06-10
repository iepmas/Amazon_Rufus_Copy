# AI Agent for a Commerce Website

## Overview

This project is an exercise in designing and deploying a multimodal AI agent capable of assisting users in a commerce setting. Inspired by [Amazon Rufus](https://www.aboutamazon.com/news/retail/amazon-rufus), the agent supports:

- General conversation
- Text-based product recommendations
- Image-based product search

All search and recommendation results are based on a predefined product catalog. The catalog is located at `data/styles.csv`, which contains metadata for various clothing items. Associated product images were downloaded locally and used to generate vector embeddings.

To simplify deployment and avoid the need for a full-scale database system, the catalog is stored as a `.csv` file. This assumes that real-world commerce platforms could export product data from relational databases in a similar format.

> Note: Due to time constraints related to my convocation and family arriving this week, only core features have been implemented in this version.

## Features

- Conversational Interface: Responds to general-purpose queries such as “What can you do?” or “What’s your name?”
- Text-Based Recommendations: Answers queries like “Show me shoes for running” and returns relevant catalog items.
- Image-Based Search: Accepts an uploaded image and returns visually similar products from the catalog.

## Tech Stack

### Frontend

- React with Vite
- Tailwind CSS
- Deployed using Vercel

#### Rationale

React with Vite was chosen for its fast development experience and hot module reload support. Tailwind CSS enabled quick prototyping without extensive styling overhead. As this was my first time working with Tailwind, this project provided an opportunity to experiment with its utility-first design approach. The frontend is implemented as a single-page application focused on usability and minimal friction.

### Backend

- FastAPI (Python)
- LangChain with OpenAI APIs for retrieval-augmented generation
- FAISS for vector-based similarity search
- ResNet18 model for image embeddings
- Deployed using Heroku

#### Rationale

FastAPI was selected for its speed and modern Pythonic interface. LangChain simplified the logic needed to integrate LLM-driven conversation with semantic retrieval. FAISS allowed efficient vector searches even on limited hardware. Initially, OpenAI’s CLIP was considered for image embeddings, but due to memory and storage constraints on the Heroku free tier, I opted for a lighter solution using ResNet18. I also reduced the FAISS index size (from ~44,000 entries to ~2,000) and downsampled the images to reduce resource usage.

## Data Pipeline

- The text-based agent was tested using two datasets: `laptop_prices.csv` and `styles.csv`. Both were imported into a local MySQL database and queried to populate the FAISS text index.
- The image-based agent was tested using product images associated with `styles.csv`.
- Fashion metadata comes from the [`styles.csv`](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) file, available via Kaggle. Images can also be downloaded and unzipped into the `data/` folder.
- Laptop price metadata comes from the [`laptop_prices.csv`](https://www.kaggle.com/datasets/owm4096/laptop-prices) dataset.
- Embeddings (text and image) are precomputed and stored in FAISS indexes to enable fast similarity search.

## How to Generate Image and Text Embeddings

1. Place the product images in the `data/images/` directory. Image file names should match the `id` column in `styles.csv` (e.g., `12345.jpg`).
2. Run the image + text embedding script:
   ```bash
   python scripts/load_image_embeddings.py
   python scripts/build_text_faiss.py
   ```
## Running the Project

### 1. Backend (FastAPI)

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Make sure your environment variables (e.g., `OPENAI_API_KEY`) are set in a `.env` file in the project root.

Then run:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

This will launch the API server locally at `http://127.0.0.1:8080`.

### 2. Frontend (Vite/React)

In the `frontend/` directory, create a `.env` file with:
```env
VITE_API_BASE_URL=http://127.0.0.1:8080
```

Then run:
```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

| Endpoint             | Method | Description                                                   |
|----------------------|--------|---------------------------------------------------------------|
| `/api/text-search`   | POST   | Accepts a user query and returns top matching product results |
| `/api/image-search`  | POST   | Accepts an uploaded image and returns visually similar items  |

All endpoints return JSON responses and are CORS-enabled for frontend use.

## Additional Notes

Due to the limited time available this week—especially as my convocation is taking place and my parents are visiting—I focused on delivering a functional prototype with clean end-to-end flow. If I had more time, I would:

- Combine the text and image agents into a unified pipeline
- Improve the frontend to display product cards with image, price, and metadata
- Expand Swagger/OpenAPI documentation
- Add automated testing and deploy to a more scalable backend (e.g., Railway or Fly.io)
- Couldn't locate dataset containing images and prices. However, with the OpenAI model's ability to reason, I believe if there were a price field that the model should perform adequately. Verified this with another dataset containing prices but no images.
