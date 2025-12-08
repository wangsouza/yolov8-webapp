import asyncio
from typing import List, Optional
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import time
import numpy as np
import cv2
from ultralytics import YOLO

TARGET_CLASSES_DEFAULT = ["person", "laptop", "cell phone", "cup", "book", "mug"]

class InferenceConfig(BaseModel):
    confidence: float = 0.75
    iou: float = 0.5
    classes: List[str] = TARGET_CLASSES_DEFAULT

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("yolov8n.pt")
MODEL_NAMES = model.model.names if hasattr(model, "model") else {}

# Serve frontend index and static files
FRONTEND_DIR = (Path(__file__).resolve().parent.parent / "frontend").resolve()

@app.get("/")
async def index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("OK")

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    config = InferenceConfig()  # default
    busy = False
    last_proc_ts = 0.0
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "config":
                try:
                    config = InferenceConfig(**msg.get("payload", {}))
                    await ws.send_json({"type": "log", "message": f"Config updated: {config.dict()}"})
                except Exception as e:
                    try:
                        await ws.send_json({"type": "error", "message": str(e)})
                    except WebSocketDisconnect:
                        break
            elif msg.get("type") == "frame":
                # Expect base64 JPEG frame
                b64 = msg.get("payload", {}).get("image")
                if not b64:
                    try:
                        await ws.send_json({"type": "error", "message": "No image in payload"})
                    except WebSocketDisconnect:
                        break
                    continue
                try:
                    # Simple rate limit: process at ~10 FPS
                    now = time.time()
                    if busy or (now - last_proc_ts) < 0.08:
                        continue
                    busy = True
                    import base64
                    data = base64.b64decode(b64)
                    arr = np.frombuffer(data, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    # Run inference on CPU, restrict classes and set IOU
                    selected_ids = [cid for cid, name in MODEL_NAMES.items() if name in config.classes]
                    res = model.predict(
                        source=img,
                        conf=config.confidence,
                        iou=config.iou,
                        device="cpu",
                        classes=selected_ids if selected_ids else None,
                    )
                    names = MODEL_NAMES
                    detections = []
                    r0 = res[0]
                    if hasattr(r0, "boxes") and r0.boxes is not None:
                        for box in r0.boxes:
                            cls_id = int(box.cls[0])
                            cls_name = names.get(cls_id, str(cls_id))
                            conf = float(box.conf[0])
                            if cls_name in config.classes and conf >= config.confidence:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                detections.append({
                                    "cls": cls_name,
                                    "conf": conf,
                                    "xyxy": [x1, y1, x2, y2]
                                })
                    try:
                        await ws.send_json({"type": "detections", "payload": {"detections": detections}})
                    except WebSocketDisconnect:
                        break
                    finally:
                        busy = False
                        last_proc_ts = time.time()
                except Exception as e:
                    busy = False
                    try:
                        await ws.send_json({"type": "error", "message": str(e)})
                    except WebSocketDisconnect:
                        break
            else:
                try:
                    await ws.send_json({"type": "log", "message": f"Unknown message type: {msg.get('type')}"})
                except WebSocketDisconnect:
                    break
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
