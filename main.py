import io
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="Image Defect Detection API")

MODEL_PATH = Path(__file__).with_name("trained.pt")
_model = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        _model = YOLO(str(MODEL_PATH))
    return _model


def build_detection_response(results: list[Any], filename: str) -> dict[str, Any]:
    defects: list[dict[str, Any]] = []

    for result in results:
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if not boxes:
            continue

        for box in boxes:
            confidence = 0.0
            if hasattr(box, "conf") and box.conf is not None:
                conf_values = box.conf.tolist() if hasattr(box.conf, "tolist") else box.conf
                if conf_values:
                    confidence = float(conf_values[0])

            class_id = None
            if hasattr(box, "cls") and box.cls is not None:
                class_values = box.cls.tolist() if hasattr(box.cls, "tolist") else box.cls
                if class_values:
                    class_id = int(class_values[0])

            label = names.get(class_id, "unknown") if class_id is not None else "unknown"

            xyxy = None
            if hasattr(box, "xyxy") and box.xyxy is not None:
                xyxy_values = box.xyxy.tolist() if hasattr(box.xyxy, "tolist") else box.xyxy
                if xyxy_values:
                    xyxy = [float(value) for value in xyxy_values[0]]

            defects.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "bounding_box": xyxy,
                }
            )

    return {
        "filename": filename,
        "has_defects": bool(defects),
        "defect_count": len(defects),
        "defects": defects,
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/detect")
async def detect_defects(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        with Image.open(io.BytesIO(contents)) as image:
            image.verify()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc

    file_extension = Path(file.filename).suffix.lower() or ".png"
    with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
        temp_file.write(contents)
        temp_path = temp_file.name

    try:
        model = get_model()
        results = model(temp_path, save=False)
        response = build_detection_response(results, file.filename)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
