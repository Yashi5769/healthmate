import cv2
import torch
import numpy as np
import time
import json
import asyncio
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from datetime import datetime
from typing import List

# --- YOLOv7 & Fall Detection Logic ---
# Note: Adjust imports based on your specific repo structure
# Assuming standard YOLOv7 imports are available or copied from your existing video.py
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

class FallDetector:
    def __init__(self, weights='yolov7-w6-pose.pt', source='0', device='cpu'):
        self.device = select_device(device)
        self.model = attempt_load(weights, map_location=self.device)
        self.model.eval()
        
        # Open Webcam
        self.cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open webcam source: {source}")
            
        # State
        self.current_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.fall_count = 0
        
        # Simple Tracker State
        self.next_track_id = 1
        self.tracks = {} # {id: {'center': (x,y), 'lost_count': 0}}
        
        # Start capture thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Inference Logic
            processed_frame, falls = self.process_frame(frame)
            
            with self.lock:
                self.current_frame = processed_frame
                self.latest_falls = falls
            
            time.sleep(0.01) # Prevent CPU spin

    def process_frame(self, frame):
        # Preprocess
        img = cv2.resize(frame, (640, 640))
        img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Detect
        falls_detected = []
        with torch.no_grad():
            pred = self.model(img)[0]
            # Apply NMS
            pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        # Process Detections
        # Note: This is a simplified fallback if Pose logic isn't fully integrated
        # You should replace this with your specific keypoint analysis from video.py
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                
                for *xyxy, conf, cls in reversed(det):
                    # Heuristic: If Aspect Ratio (Width/Height) > 1.2, possibly fallen
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    w = x2 - x1
                    h = y2 - y1
                    aspect_ratio = w / h
                    
                    label = f'{self.model.names[int(cls)]} {conf:.2f}'
                    color = (255, 0, 0)
                    
                    is_fall = False
                    # Basic Fall Logic (Replace with Keypoint logic for accuracy)
                    if aspect_ratio > 1.2 and h < w: 
                        label += " FALL DETECTED"
                        color = (0, 0, 255)
                        is_fall = True
                        
                    plot_one_box(xyxy, frame, label=label, color=color, line_thickness=2)
                    
                    if is_fall:
                        # Assign simple ID based on proximity (Simulated Tracking)
                        tid = self.get_track_id(x1+w/2, y1+h/2)
                        
                        falls_detected.append({
                            "person_tracking_id": tid,
                            "bbox": {"x": x1, "y": y1, "width": w, "height": h},
                            "confidence": float(conf)
                        })

        return frame, falls_detected

    def get_track_id(self, cx, cy):
        # Very naive tracker for demo purposes
        min_dist = 100.0
        found_id = None
        
        # Clean old tracks
        ids_to_remove = []
        for tid, data in self.tracks.items():
            data['lost_count'] += 1
            if data['lost_count'] > 30: # 1 sec @ 30fps
                ids_to_remove.append(tid)
            else:
                dist = np.sqrt((cx - data['center'][0])**2 + (cy - data['center'][1])**2)
                if dist < min_dist:
                    min_dist = dist
                    found_id = tid
        
        for tid in ids_to_remove:
            del self.tracks[tid]
            
        if found_id is not None:
            self.tracks[found_id]['center'] = (cx, cy)
            self.tracks[found_id]['lost_count'] = 0
            return found_id
        else:
            new_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[new_id] = {'center': (cx, cy), 'lost_count': 0}
            return new_id

    def get_frame(self):
        with self.lock:
            if self.current_frame is None:
                return None
            return self.current_frame.copy()

    def get_alerts(self):
        with self.lock:
            return list(self.latest_falls)

# --- FastAPI Server ---

app = FastAPI()

# Allow CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Detector (Global Singleton)
detector = None

@app.on_event("startup")
def startup_event():
    global detector
    # Load default weights, ensure 'yolov7-w6-pose.pt' exists or change path
    try:
        detector = FallDetector(weights='yolov7-w6-pose.pt', source='0')
        print("Model loaded and camera started.")
    except Exception as e:
        print(f"Error starting detector: {e}")

# Connection Manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

# Background Task to Push Alerts
async def alert_broadcaster():
    last_alert_time = {} # Throttle alerts per tracking_id
    
    while True:
        await asyncio.sleep(0.1)
        if detector and detector.latest_falls:
            current_time = time.time()
            
            for fall in detector.latest_falls:
                tid = fall['person_tracking_id']
                
                # Throttle: 1 alert every 3 seconds per person
                if tid not in last_alert_time or (current_time - last_alert_time[tid] > 3.0):
                    last_alert_time[tid] = current_time
                    detector.fall_count += 1
                    
                    # Construct message matching React's FallAlertMessage
                    message = {
                        "type": "fall_detected",
                        "data": {
                            "patient_id": "room-1", # You can make this dynamic
                            "person_tracking_id": tid,
                            "fall_count": detector.fall_count,
                            "timestamp": datetime.utcnow().isoformat(),
                            "metadata": {
                                "bounding_box": fall['bbox'],
                                "confidence": fall['confidence']
                            }
                        }
                    }
                    await manager.broadcast(json.dumps(message))

# Start broadcaster on startup
@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(alert_broadcaster())

# --- API Endpoints ---

@app.get("/api/video/stream")
def video_feed(patient_id: str = "default"):
    """MJPEG Video Stream for React Frontend"""
    def iterfile():
        while True:
            if detector:
                frame = detector.get_frame()
                if frame is not None:
                    _, buffer = cv2.imencode('.jpg', frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.04) # ~25 FPS

    return StreamingResponse(iterfile(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/api/ws/alerts")
async def websocket_endpoint(websocket: WebSocket, patient_id: str = "default"):
    """WebSocket for Real-time Alerts"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/stats")
def get_stats():
    """Optional stats endpoint"""
    return {
        "fps": 30.0,
        "is_processing": detector.running if detector else False,
        "fall_count": detector.fall_count if detector else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)