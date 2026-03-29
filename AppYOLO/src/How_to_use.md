# AppYOLO Web Quick Start

## 1) Install Dependencies

```bash
pip install -r requirements.txt
```

## 2) Start Backend + Frontend

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## 3) Open Dashboard

Open this URL in your browser:

```text
http://127.0.0.1:8000
```

## API List

- `GET /api/health`
    - Backend health, model status, live worker status.
- `GET /api/model/info`
    - Model path and class mapping from YOLO.
- `POST /api/inference/image`
    - Multipart image upload inference.
    - Returns detections, decision, explainability, temperature, and annotated image base64.
- `POST /api/inference/local`
    - Inference using a workspace-relative local image path.
- `POST /api/inference/video`
    - Multipart video upload inference.
    - Runs original video logic in headless mode and outputs processed video file.
- `POST /api/inference/video/local`
    - Inference using a workspace-relative local video path.
- `POST /api/pipeline/main/run`
    - Runs the original main workflow logic through backend API.
- `POST /api/pipeline/vcn/run`
    - Runs the original VCN multi-camera + map composition logic through backend API.
- `GET /api/generated/files`
    - Lists generated files under outputs for frontend gallery rendering.
- `POST /api/live/start`
    - Start live stream worker (source `0` for webcam, or a video path).
- `POST /api/live/stop`
    - Stop live stream worker.
- `GET /api/live/state`
    - Current live state and recent history arrays for charting.
- `GET /api/live/frame`
    - Latest annotated JPEG frame.
- `GET /api/live/events`
    - Server Sent Events stream for dynamic metric updates.

## Minimal Curl Example

```bash
curl -X POST "http://127.0.0.1:8000/api/live/start" \
    -H "Content-Type: application/json" \
    -d '{"source":"0","conf":0.25}'
```