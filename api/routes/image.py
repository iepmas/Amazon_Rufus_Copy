from fastapi import APIRouter, UploadFile, File, HTTPException
from agents.image_search import search_similar_images, init_image_search
import tempfile
import shutil

router = APIRouter()

index, metadata, model, preprocess = init_image_search()

@router.post("/image-search")
async def image_search(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        results = search_similar_images(tmp_path, index, metadata, model, preprocess)
        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
