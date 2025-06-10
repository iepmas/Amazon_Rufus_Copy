from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.text import router as text_router
from api.routes.image import router as image_router

app = FastAPI(
    title="Raymond",
    description="A multimodal shopping assistant with text and image-based search",
    version="1.0.0"
)

origins = [
    "http://localhost:5173",
    "https://amazon-rufus-copy.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include text and image search routers
app.include_router(text_router, prefix="/api")
app.include_router(image_router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Hello World"}
