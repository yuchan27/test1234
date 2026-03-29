from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Optional

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

from src.inference.infer import YOLOInfer
from src.pipeline_service import run_main_pipeline, run_vcn_pipeline
from src.inference.utils import convert_to_yolo_format


ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
OUTPUT_DIR = ROOT_DIR / "outputs"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
OUTPUT_VIDEO_DIR = OUTPUT_DIR / "videos"
LIVE_LOG_DIR = OUTPUT_DIR / "live_logs"
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
LIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)

INFER_LOCK = threading.Lock()
ROOM_TEMP_TARGET_C = 25.0
ROOM_TEMP_MIN_C = 22.0
ROOM_TEMP_MAX_C = 28.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _pick_model_path() -> Path:
    candidates = [
        ROOT_DIR / "models" / "release11.pt",
        ROOT_DIR / "models" / "release26.pt",
        ROOT_DIR / "models" / "release.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No model weight found under models/.")


MODEL_PATH: Optional[Path]
MODEL_LOAD_ERROR: str = ""
INFER_ENGINE: Optional[YOLOInfer]

try:
    MODEL_PATH = _pick_model_path()
    INFER_ENGINE = YOLOInfer(str(MODEL_PATH))
except Exception as exc:  # pragma: no cover
    MODEL_PATH = None
    INFER_ENGINE = None
    MODEL_LOAD_ERROR = str(exc)


@dataclass
class LiveTracker:
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None
    running: bool = False
    source: str = "0"
    last_error: str = ""
    latest_metrics: dict[str, Any] = field(default_factory=dict)
    latest_jpeg: Optional[bytes] = None
    risk_history: Deque[float] = field(default_factory=lambda: deque(maxlen=180))
    temp_history: Deque[float] = field(default_factory=lambda: deque(maxlen=180))
    frame_id_history: Deque[int] = field(default_factory=lambda: deque(maxlen=180))
    timestamp_history: Deque[str] = field(default_factory=lambda: deque(maxlen=180))
    current_log_path: str = ""
    system_temp_offset: Optional[float] = None
    normalized_system_temp: float = ROOM_TEMP_TARGET_C
    raw_system_temp: Optional[float] = None
    system_temp_source: str = "unavailable"


LIVE_TRACKER = LiveTracker()


class LocalImageRequest(BaseModel):
    image_path: str = Field(..., description="Relative path under workspace")
    save_annotated: bool = True


class LocalVideoRequest(BaseModel):
    video_path: str = Field(..., description="Relative path under workspace")
    output_video_path: str = Field(default="outputs/out.mp4", description="Output path under workspace")
    with_decision: bool = True


class MainPipelineRequest(BaseModel):
    image_path: str = "test/left1.png"
    video_path: str = "test/dataset/forest1.avi"
    output_video_path: str = "outputs/out.mp4"
    with_decision: bool = True


class VCNPipelineRequest(BaseModel):
    package_dir: str = "WebCamPackage"
    map_json_path: str = "WebCamPackage/_map.json"


class LiveStartRequest(BaseModel):
    source: str = Field(default="0", description="Camera index or video path")
    conf: float = Field(default=0.25, ge=0.05, le=0.95)
    frame_skip: int = Field(default=1, ge=1, le=8)
    max_frame_width: int = Field(default=1280, ge=640, le=3840)


def _ensure_engine() -> YOLOInfer:
    if INFER_ENGINE is None:
        reason = MODEL_LOAD_ERROR or "model not initialized"
        raise HTTPException(status_code=500, detail=f"Model unavailable: {reason}")
    return INFER_ENGINE


def _resolve_workspace_path(input_path: str) -> Path:
    candidate = Path(input_path)
    if not candidate.is_absolute():
        candidate = (ROOT_DIR / candidate).resolve()
    else:
        candidate = candidate.resolve()

    try:
        candidate.relative_to(ROOT_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Path must be inside workspace") from exc

    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Path does not exist")

    return candidate


def _resolve_output_path(input_path: str) -> Path:
    candidate = Path(input_path)
    if not candidate.is_absolute():
        candidate = (ROOT_DIR / candidate).resolve()
    else:
        candidate = candidate.resolve()

    try:
        candidate.relative_to(ROOT_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Output path must be inside workspace") from exc

    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _to_relative_workspace_path(path: Path) -> str:
    return str(path.relative_to(ROOT_DIR)).replace("\\", "/")


def _to_output_url(path: Path) -> str:
    rel = path.relative_to(OUTPUT_DIR).as_posix()
    return f"/outputs/{rel}"


def _guess_file_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in ALLOWED_IMAGE_SUFFIXES:
        return "image"
    if suffix in ALLOWED_VIDEO_SUFFIXES:
        return "video"
    if suffix in {".json", ".txt", ".log"}:
        return "text"
    return "other"


def _list_generated_files(limit: int = 80) -> list[dict[str, Any]]:
    files = [path for path in OUTPUT_DIR.rglob("*") if path.is_file()]
    files.sort(key=lambda item: item.stat().st_mtime, reverse=True)

    payload = []
    for path in files[:limit]:
        stat = path.stat()
        payload.append(
            {
                "name": path.name,
                "kind": _guess_file_kind(path),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "relative_path": f"outputs/{path.relative_to(OUTPUT_DIR).as_posix()}",
                "url": _to_output_url(path),
            }
        )

    return payload


def _cleanup_generated_files(keep_latest: int = 40) -> list[str]:
    files = [path for path in OUTPUT_DIR.rglob("*") if path.is_file()]
    files.sort(key=lambda item: item.stat().st_mtime, reverse=True)

    if keep_latest < 0:
        keep_latest = 0

    to_delete = files[keep_latest:]
    deleted: list[str] = []
    for path in to_delete:
        try:
            path.unlink()
            deleted.append(f"outputs/{path.relative_to(OUTPUT_DIR).as_posix()}")
        except OSError:
            continue

    return deleted


def _extract_video_preview_base64(video_path: Path) -> str:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return ""

    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            return ""

        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            return ""

        return base64.b64encode(encoded.tobytes()).decode("ascii")
    finally:
        cap.release()


def _build_video_telemetry(video_results: list[dict[str, Any]], max_points: int = 160) -> dict[str, Any]:
    if not video_results:
        return {
            "points": [],
            "max_risk": 0.0,
            "avg_risk": 0.0,
            "max_temperature": 0.0,
            "avg_temperature": 0.0,
        }

    sample_step = max(1, (len(video_results) + max_points - 1) // max_points)
    points = []

    risk_values: list[float] = []
    temp_values: list[float] = []

    for idx, item in enumerate(video_results):
        decision_full = item.get("decision", {})
        decision = decision_full.get("decision", {}) if isinstance(decision_full, dict) else {}

        risk_score = float(decision.get("risk_score", 0.0) or 0.0)
        vision_temp = item.get("vision_temp")
        temp_value = float(vision_temp) if vision_temp is not None else 0.0
        detection_count = len(item.get("detections", []) or [])

        risk_values.append(risk_score)
        temp_values.append(temp_value)

        if idx % sample_step == 0 or idx == len(video_results) - 1:
            frame_id_val = int(item.get("frame_id", idx))
            points.append(
                {
                    "frame_id": frame_id_val,
                    "time_sec": round(frame_id_val / 30.0, 2),
                    "risk_score": risk_score,
                    "vision_temperature_celsius": temp_value,
                    "detection_count": detection_count,
                    "trigger_alarm": bool(decision.get("trigger_alarm", False)),
                    "suggested_action": decision.get("suggested_action", "CONTINUE_MONITORING"),
                }
            )

    return {
        "points": points,
        "max_risk": max(risk_values) if risk_values else 0.0,
        "avg_risk": (sum(risk_values) / len(risk_values)) if risk_values else 0.0,
        "max_temperature": max(temp_values) if temp_values else 0.0,
        "avg_temperature": (sum(temp_values) / len(temp_values)) if temp_values else 0.0,
    }


def _source_to_capture_arg(source: str) -> Any:
    text = source.strip()
    if text.isdigit():
        return int(text)
    return text


def _create_capture(source: str) -> cv2.VideoCapture:
    source_arg = _source_to_capture_arg(source)
    if isinstance(source_arg, int):
        cap = cv2.VideoCapture(source_arg, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(source_arg)
        return cap

    return cv2.VideoCapture(source_arg)


def _read_system_temperature_celsius() -> tuple[Optional[float], str]:
    if psutil is None or not hasattr(psutil, "sensors_temperatures"):
        return None, "unavailable"

    # Priority: real thermal sensor value (CPU/package/core)
    try:
        groups = psutil.sensors_temperatures(fahrenheit=False) or {}
    except Exception:
        groups = {}

    candidates: list[tuple[int, float]] = []
    for chip_name, entries in groups.items():
        chip_key = str(chip_name).lower()
        for entry in entries:
            current = getattr(entry, "current", None)
            if current is None:
                continue

            try:
                value = float(current)
            except (TypeError, ValueError):
                continue

            if value < -20.0 or value > 140.0:
                continue

            label_key = str(getattr(entry, "label", "") or "").lower()
            key = f"{chip_key} {label_key}"

            priority = 0
            if any(token in key for token in ("cpu", "package", "core", "k10temp", "tctl", "soc", "acpitz")):
                priority = 3
            elif any(token in key for token in ("thermal", "pch", "gpu")):
                priority = 2

            candidates.append((priority, value))

    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return round(candidates[0][1], 2), "sensor"

    # Fallback: estimate thermal trend from CPU load when direct sensors are unavailable.
    try:
        cpu_load = float(psutil.cpu_percent(interval=None))
    except Exception:
        return None, "unavailable"

    estimated_temp = 42.0 + (cpu_load * 0.18)
    return round(estimated_temp, 2), "cpu-load-estimate"


def _normalize_system_temperature(raw_temp: Optional[float], source: str) -> Optional[float]:
    if raw_temp is None:
        return None

    with LIVE_TRACKER.lock:
        LIVE_TRACKER.raw_system_temp = round(raw_temp, 2)
        LIVE_TRACKER.system_temp_source = source

        if LIVE_TRACKER.system_temp_offset is None:
            LIVE_TRACKER.system_temp_offset = raw_temp - ROOM_TEMP_TARGET_C

        normalized = raw_temp - LIVE_TRACKER.system_temp_offset
        normalized = ROOM_TEMP_TARGET_C + ((normalized - ROOM_TEMP_TARGET_C) * 0.35)
        normalized = max(ROOM_TEMP_MIN_C, min(ROOM_TEMP_MAX_C, normalized))

        LIVE_TRACKER.normalized_system_temp = round(
            (0.72 * LIVE_TRACKER.normalized_system_temp) + (0.28 * normalized),
            2,
        )
        return LIVE_TRACKER.normalized_system_temp


def _build_payload(engine: YOLOInfer, result: Any, frame_id: int, vision_temp: Optional[float]) -> dict[str, Any]:
    return {
        "context": {
            "timestamp": _utc_now_iso(),
            "frame_id": frame_id,
        },
        "perceptions": {
            "visual_objects": engine.image_infer._to_yolo_format_str(result),
            "environmental_sensors": {
                "temperature_celsius": vision_temp if vision_temp is not None else 25.0,
            },
        },
    }


def _append_live_log(log_path: str, metrics: dict[str, Any]) -> None:
    if not log_path:
        return

    row = {
        "timestamp": metrics.get("timestamp", _utc_now_iso()),
        "frame_id": int(metrics.get("frame_id", 0)),
        "risk_score": float(metrics.get("risk_score", 0.0)),
        "vision_temperature_celsius": float(metrics.get("vision_temperature_celsius") or 0.0),
        "system_temperature_celsius": metrics.get("system_temperature_celsius"),
        "system_temperature_source": metrics.get("system_temperature_source", "unavailable"),
        "fps": float(metrics.get("fps", 0.0)),
        "detection_count": int(metrics.get("detection_count", 0)),
    }

    try:
        with open(log_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        return


def _run_frame_inference(
    engine: YOLOInfer,
    frame: Any,
    frame_id: int,
    conf: float,
    max_frame_width: int,
) -> tuple[dict[str, Any], bytes]:
    frame_for_infer = frame
    if max_frame_width > 0 and frame.shape[1] > max_frame_width:
        ratio = max_frame_width / float(frame.shape[1])
        resized_h = max(int(frame.shape[0] * ratio), 1)
        frame_for_infer = cv2.resize(frame, (max_frame_width, resized_h))

    with INFER_LOCK:
        yolo_results = engine.image_infer.model(frame_for_infer, verbose=False, conf=conf)
    result = yolo_results[0]

    detections = convert_to_yolo_format(result)
    flame_temp = engine.image_infer.temp_estimator._estimate_temperature_from_frame(frame_for_infer, result)
    raw_system_temp, system_temp_source = _read_system_temperature_celsius()

    if raw_system_temp is None and flame_temp is not None:
        raw_system_temp = float(flame_temp)
        system_temp_source = "vision-fallback"

    normalized_system_temp = _normalize_system_temperature(raw_system_temp, system_temp_source)

    # Use normalized computer thermal reading when available, keeping it around room-temperature baseline.
    effective_temp = normalized_system_temp
    if effective_temp is None:
        effective_temp = ROOM_TEMP_TARGET_C

    payload = _build_payload(engine, result, frame_id, effective_temp)
    decision_full = engine.image_infer.decision_engine.evaluate_payload(payload)
    decision = decision_full.get("decision", {})

    class_breakdown = Counter(item.get("class_name", "unknown") for item in detections)

    annotated = result.plot(conf=False)
    ok, encoded = cv2.imencode(".jpg", annotated)
    if not ok:
        raise RuntimeError("failed to encode annotated frame")

    metrics = {
        "timestamp": payload["context"]["timestamp"],
        "frame_id": frame_id,
        "detections": detections,
        "detection_count": len(detections),
        "class_breakdown": dict(class_breakdown),
        "vision_temperature_celsius": effective_temp,
        "flame_temperature_celsius": flame_temp,
        "system_temperature_celsius": raw_system_temp,
        "system_temperature_source": system_temp_source,
        "decision": decision,
        "explainability": decision_full.get("explainability", {}),
    }

    return metrics, encoded.tobytes()


def _public_live_state() -> dict[str, Any]:
    with LIVE_TRACKER.lock:
        return {
            "running": LIVE_TRACKER.running,
            "source": LIVE_TRACKER.source,
            "last_error": LIVE_TRACKER.last_error,
            "current_log_path": LIVE_TRACKER.current_log_path,
            "latest_metrics": LIVE_TRACKER.latest_metrics,
            "thermal": {
                "raw_system_temperature_celsius": LIVE_TRACKER.raw_system_temp,
                "normalized_system_temperature_celsius": LIVE_TRACKER.normalized_system_temp,
                "system_temperature_source": LIVE_TRACKER.system_temp_source,
            },
            "history": {
                "timestamps": list(LIVE_TRACKER.timestamp_history),
                "frame_ids": list(LIVE_TRACKER.frame_id_history),
                "risk": list(LIVE_TRACKER.risk_history),
                "temperature": list(LIVE_TRACKER.temp_history),
            },
        }


def _stop_live_worker(timeout: float = 3.0) -> None:
    thread_to_join: Optional[threading.Thread] = None

    with LIVE_TRACKER.lock:
        if LIVE_TRACKER.thread and LIVE_TRACKER.thread.is_alive():
            LIVE_TRACKER.stop_event.set()
            thread_to_join = LIVE_TRACKER.thread

    if thread_to_join is not None:
        thread_to_join.join(timeout=timeout)

    with LIVE_TRACKER.lock:
        LIVE_TRACKER.thread = None
        LIVE_TRACKER.running = False


def _live_worker(source: str, conf: float, frame_skip: int, max_frame_width: int) -> None:
    engine = INFER_ENGINE
    if engine is None:
        with LIVE_TRACKER.lock:
            LIVE_TRACKER.running = False
            LIVE_TRACKER.last_error = "Model not ready"
        return

    cap = _create_capture(source)
    if not cap.isOpened():
        with LIVE_TRACKER.lock:
            LIVE_TRACKER.running = False
            LIVE_TRACKER.last_error = f"Unable to open source: {source}"
        return

    with LIVE_TRACKER.lock:
        LIVE_TRACKER.running = True
        LIVE_TRACKER.source = source
        LIVE_TRACKER.last_error = ""

    frame_id = 0
    raw_frame_id = 0
    consecutive_failures = 0
    max_failures_before_reopen = 20

    try:
        while not LIVE_TRACKER.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures < max_failures_before_reopen:
                    time.sleep(0.04)
                    continue

                cap.release()
                cap = _create_capture(source)
                if not cap.isOpened():
                    with LIVE_TRACKER.lock:
                        LIVE_TRACKER.last_error = "Camera read failed and reconnect did not recover"
                    break

                consecutive_failures = 0
                continue

            consecutive_failures = 0
            raw_frame_id += 1

            if frame_skip > 1 and raw_frame_id % frame_skip != 0:
                continue

            t0 = time.perf_counter()
            try:
                metrics, frame_jpeg = _run_frame_inference(
                    engine,
                    frame,
                    frame_id,
                    conf,
                    max_frame_width,
                )
            except Exception as exc:
                with LIVE_TRACKER.lock:
                    LIVE_TRACKER.last_error = f"Inference failed: {exc}"
                time.sleep(0.04)
                continue

            elapsed = max(time.perf_counter() - t0, 1e-6)
            metrics["fps"] = round(1.0 / elapsed, 2)
            metrics["risk_score"] = float(metrics.get("decision", {}).get("risk_score", 0.0))

            with LIVE_TRACKER.lock:
                LIVE_TRACKER.latest_metrics = metrics
                LIVE_TRACKER.latest_jpeg = frame_jpeg
                LIVE_TRACKER.frame_id_history.append(frame_id)
                LIVE_TRACKER.risk_history.append(metrics["risk_score"])
                LIVE_TRACKER.timestamp_history.append(metrics.get("timestamp", _utc_now_iso()))
                temp = metrics.get("vision_temperature_celsius")
                LIVE_TRACKER.temp_history.append(float(temp) if temp is not None else 0.0)
                log_path = LIVE_TRACKER.current_log_path

            _append_live_log(log_path, metrics)

            frame_id += 1
    finally:
        cap.release()
        with LIVE_TRACKER.lock:
            LIVE_TRACKER.running = False


app = FastAPI(
    title="AppYOLO Fire Command API",
    version="1.0.0",
    description="YOLO inference backend with dynamic dashboard APIs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")

if OUTPUT_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/")
def index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)


@app.get("/player")
def player_page() -> FileResponse:
    player_path = FRONTEND_DIR / "player.html"
    if not player_path.exists():
        raise HTTPException(status_code=404, detail="Video player not found")
    return FileResponse(player_path)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if INFER_ENGINE is not None else "degraded",
        "model_ready": INFER_ENGINE is not None,
        "model_path": str(MODEL_PATH) if MODEL_PATH else "",
        "model_error": MODEL_LOAD_ERROR,
        "live_running": LIVE_TRACKER.running,
        "timestamp": _utc_now_iso(),
    }


@app.get("/api/model/info")
def model_info() -> dict[str, Any]:
    engine = _ensure_engine()
    names = engine.image_infer.model.names
    return {
        "status": "ready",
        "model_path": str(MODEL_PATH) if MODEL_PATH else "",
        "classes": names,
    }


@app.get("/api/generated/files")
def generated_files(limit: int = 80) -> dict[str, Any]:
    safe_limit = min(max(limit, 1), 200)
    return {
        "status": "success",
        "files": _list_generated_files(safe_limit),
    }


@app.get("/api/video/thumbnail")
def video_thumbnail(path: str) -> Response:
    video_path = _resolve_workspace_path(path)
    if video_path.suffix.lower() not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only common video formats are supported")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Unable to open video")

    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise HTTPException(status_code=500, detail="Unable to read video frame")

        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise HTTPException(status_code=500, detail="Unable to encode preview frame")

        return Response(content=encoded.tobytes(), media_type="image/jpeg")
    finally:
        cap.release()


@app.get("/api/video/mjpeg")
def video_mjpeg(path: str, fps: int = 15, loop: bool = True) -> StreamingResponse:
    video_path = _resolve_workspace_path(path)
    if video_path.suffix.lower() not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only common video formats are supported")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Unable to open video")

    interval = 1.0 / max(1, min(fps, 30))

    def frame_generator() -> Any:
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
                )
                time.sleep(interval)
        finally:
            cap.release()

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=headers,
    )


@app.post("/api/generated/files/cleanup")
def generated_files_cleanup(keep_latest: int = 40) -> dict[str, Any]:
    safe_keep = min(max(keep_latest, 0), 200)
    deleted = _cleanup_generated_files(keep_latest=safe_keep)
    return {
        "status": "success",
        "keep_latest": safe_keep,
        "deleted_count": len(deleted),
        "deleted_files": deleted,
        "files": _list_generated_files(limit=80),
    }


@app.post("/api/inference/local")
def inference_local(payload: LocalImageRequest) -> dict[str, Any]:
    engine = _ensure_engine()
    path = _resolve_workspace_path(payload.image_path)

    if path.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only jpg/jpeg/png are supported")

    frame_id = int(time.time() * 1000) % 1_000_000_000
    try:
        with INFER_LOCK:
            detections, decision_full, vision_temp = engine.run_with_decision(
                image_path=str(path),
                save=payload.save_annotated,
                frame_id=frame_id,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    class_breakdown = Counter(item.get("class_name", "unknown") for item in detections)

    annotated_b64 = ""
    annotated_rel_path = ""
    annotated_url = ""
    if payload.save_annotated:
        annotated_file = OUTPUT_IMAGE_DIR / path.name
        if annotated_file.exists():
            annotated_rel_path = _to_relative_workspace_path(annotated_file)
            annotated_url = _to_output_url(annotated_file)
            annotated_b64 = base64.b64encode(annotated_file.read_bytes()).decode("ascii")

    return {
        "status": "success",
        "inference": {
            "timestamp": _utc_now_iso(),
            "frame_id": frame_id,
            "source": _to_relative_workspace_path(path),
            "detections": detections,
            "detection_count": len(detections),
            "class_breakdown": dict(class_breakdown),
            "vision_temperature_celsius": vision_temp,
            "decision": decision_full.get("decision", {}),
            "explainability": decision_full.get("explainability", {}),
            "annotated_image_path": annotated_rel_path,
            "annotated_image_url": annotated_url,
            "annotated_image_base64": annotated_b64,
        },
    }


@app.post("/api/inference/image")
async def inference_image(file: UploadFile = File(...), save_annotated: bool = True) -> dict[str, Any]:
    engine = _ensure_engine()

    suffix = Path(file.filename or "upload.jpg").suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only jpg/jpeg/png are supported")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    request_id = uuid.uuid4().hex
    temp_name = f"{request_id}{suffix}"
    temp_path = UPLOAD_DIR / temp_name
    temp_path.write_bytes(content)

    frame_id = int(time.time() * 1000) % 1_000_000_000

    try:
        with INFER_LOCK:
            detections, decision_full, vision_temp = engine.run_with_decision(
                image_path=str(temp_path),
                save=save_annotated,
                frame_id=frame_id,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass

    class_breakdown = Counter(item.get("class_name", "unknown") for item in detections)

    annotated_b64 = ""
    annotated_rel_path = ""
    annotated_url = ""
    if save_annotated:
        annotated_file = OUTPUT_IMAGE_DIR / temp_name
        if annotated_file.exists():
            annotated_rel_path = _to_relative_workspace_path(annotated_file)
            annotated_url = _to_output_url(annotated_file)
            annotated_b64 = base64.b64encode(annotated_file.read_bytes()).decode("ascii")

    return {
        "status": "success",
        "inference": {
            "timestamp": _utc_now_iso(),
            "frame_id": frame_id,
            "source": "upload",
            "detections": detections,
            "detection_count": len(detections),
            "class_breakdown": dict(class_breakdown),
            "vision_temperature_celsius": vision_temp,
            "decision": decision_full.get("decision", {}),
            "explainability": decision_full.get("explainability", {}),
            "annotated_image_path": annotated_rel_path,
            "annotated_image_url": annotated_url,
            "annotated_image_base64": annotated_b64,
        },
    }


@app.post("/api/inference/video/local")
def inference_video_local(payload: LocalVideoRequest) -> dict[str, Any]:
    engine = _ensure_engine()
    video_path = _resolve_workspace_path(payload.video_path)
    output_path = _resolve_output_path(payload.output_video_path)

    if video_path.suffix.lower() not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only common video formats are supported")

    try:
        with INFER_LOCK:
            video_results = engine.run(
                str(video_path),
                save_path=str(output_path),
                with_decision=payload.with_decision,
                display=False,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video inference failed: {exc}") from exc

    alarm_frames = 0
    if payload.with_decision:
        for item in video_results:
            decision_full = item.get("decision", {})
            if decision_full.get("decision", {}).get("trigger_alarm", False):
                alarm_frames += 1

    telemetry = _build_video_telemetry(video_results)
    preview_b64 = _extract_video_preview_base64(output_path)
    player_url = f"/player?path={_to_relative_workspace_path(output_path)}"

    return {
        "status": "success",
        "video": {
            "source": _to_relative_workspace_path(video_path),
            "output_video_path": _to_relative_workspace_path(output_path),
            "output_video_url": _to_output_url(output_path),
            "player_url": player_url,
            "frame_count": len(video_results),
            "with_decision": payload.with_decision,
            "alarm_frame_count": alarm_frames,
            "sample": video_results[0] if video_results else {},
            "telemetry": telemetry,
            "preview_image_base64": preview_b64,
        },
    }


@app.post("/api/inference/video")
async def inference_video_upload(
    file: UploadFile = File(...),
    with_decision: bool = True,
) -> dict[str, Any]:
    engine = _ensure_engine()

    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only common video formats are supported")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    request_id = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{request_id}{suffix}"
    output_path = OUTPUT_VIDEO_DIR / f"{request_id}_out.mp4"
    upload_path.write_bytes(content)

    try:
        with INFER_LOCK:
            video_results = engine.run(
                str(upload_path),
                save_path=str(output_path),
                with_decision=with_decision,
                display=False,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video inference failed: {exc}") from exc
    finally:
        try:
            upload_path.unlink()
        except FileNotFoundError:
            pass

    alarm_frames = 0
    if with_decision:
        for item in video_results:
            decision_full = item.get("decision", {})
            if decision_full.get("decision", {}).get("trigger_alarm", False):
                alarm_frames += 1

    telemetry = _build_video_telemetry(video_results)
    preview_b64 = _extract_video_preview_base64(output_path)
    player_url = f"/player?path={_to_relative_workspace_path(output_path)}"

    return {
        "status": "success",
        "video": {
            "source": "upload",
            "output_video_path": _to_relative_workspace_path(output_path),
            "output_video_url": _to_output_url(output_path),
            "player_url": player_url,
            "frame_count": len(video_results),
            "with_decision": with_decision,
            "alarm_frame_count": alarm_frames,
            "sample": video_results[0] if video_results else {},
            "telemetry": telemetry,
            "preview_image_base64": preview_b64,
        },
    }


@app.post("/api/pipeline/main/run")
def pipeline_main_run(payload: MainPipelineRequest) -> dict[str, Any]:
    engine = _ensure_engine()

    image_path = _resolve_workspace_path(payload.image_path)
    video_path = _resolve_workspace_path(payload.video_path)
    output_video_path = _resolve_output_path(payload.output_video_path)

    try:
        with INFER_LOCK:
            result = run_main_pipeline(
                infer=engine,
                image_path=str(image_path),
                video_path=str(video_path),
                output_video_path=str(output_video_path),
                with_decision=payload.with_decision,
                display=False,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Main pipeline failed: {exc}") from exc

    return {
        "status": "success",
        "pipeline": {
            "name": "main",
            "result": result,
            "output_video_url": _to_output_url(output_video_path),
            "generated_files": _list_generated_files(limit=60),
        },
    }


@app.post("/api/pipeline/vcn/run")
def pipeline_vcn_run(payload: VCNPipelineRequest) -> dict[str, Any]:
    engine = _ensure_engine()

    package_dir = _resolve_workspace_path(payload.package_dir)
    map_json_path = _resolve_workspace_path(payload.map_json_path)

    try:
        with INFER_LOCK:
            result = run_vcn_pipeline(
                infer=engine,
                package_dir=str(package_dir),
                map_json_path=str(map_json_path),
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"VCN pipeline failed: {exc}") from exc

    output_path = _resolve_workspace_path(result["output_path"])

    camera_results = {}
    for name, camera in result.get("camera_results", {}).items():
        camera_copy = dict(camera)
        annotated_path_str = camera_copy.get("annotated_path")
        if annotated_path_str:
            annotated_path = _resolve_workspace_path(annotated_path_str)
            camera_copy["annotated_url"] = _to_output_url(annotated_path)
        camera_results[name] = camera_copy

    return {
        "status": "success",
        "pipeline": {
            "name": "vcn",
            "output_path": _to_relative_workspace_path(output_path),
            "output_url": _to_output_url(output_path),
            "processed_files": result.get("processed_files", []),
            "escape_texts": result.get("escape_texts", {}),
            "camera_results": camera_results,
            "generated_files": _list_generated_files(limit=60),
        },
    }


@app.post("/api/live/start")
def live_start(request: LiveStartRequest) -> dict[str, Any]:
    _ensure_engine()
    _stop_live_worker()

    worker = threading.Thread(
        target=_live_worker,
        args=(request.source, request.conf, request.frame_skip, request.max_frame_width),
        daemon=True,
        name="live-inference-worker",
    )

    with LIVE_TRACKER.lock:
        LIVE_TRACKER.stop_event = threading.Event()
        LIVE_TRACKER.source = request.source
        LIVE_TRACKER.latest_metrics = {}
        LIVE_TRACKER.latest_jpeg = None
        LIVE_TRACKER.last_error = ""
        LIVE_TRACKER.current_log_path = str(
            LIVE_LOG_DIR / f"live_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        LIVE_TRACKER.timestamp_history.clear()
        LIVE_TRACKER.frame_id_history.clear()
        LIVE_TRACKER.risk_history.clear()
        LIVE_TRACKER.temp_history.clear()
        LIVE_TRACKER.system_temp_offset = None
        LIVE_TRACKER.normalized_system_temp = ROOM_TEMP_TARGET_C
        LIVE_TRACKER.raw_system_temp = None
        LIVE_TRACKER.system_temp_source = "unavailable"
        LIVE_TRACKER.thread = worker

    worker.start()

    return {
        "status": "started",
        "source": request.source,
        "confidence_threshold": request.conf,
        "frame_skip": request.frame_skip,
        "max_frame_width": request.max_frame_width,
    }


@app.post("/api/live/stop")
def live_stop() -> dict[str, str]:
    _stop_live_worker()
    return {"status": "stopped"}


@app.get("/api/live/state")
def live_state() -> dict[str, Any]:
    return _public_live_state()


@app.get("/api/live/frame")
def live_frame() -> Response:
    with LIVE_TRACKER.lock:
        frame = LIVE_TRACKER.latest_jpeg

    if frame is None:
        raise HTTPException(status_code=404, detail="No live frame available")

    return Response(content=frame, media_type="image/jpeg")


@app.get("/api/live/events")
async def live_events() -> StreamingResponse:
    async def event_generator() -> Any:
        last_frame_id = -1

        try:
            while True:
                state = _public_live_state()
                frame_id = state.get("latest_metrics", {}).get("frame_id", -1)

                if frame_id != last_frame_id:
                    yield f"data: {json.dumps(state, ensure_ascii=False)}\\n\\n"
                    last_frame_id = frame_id
                elif not state.get("running", False):
                    yield f"data: {json.dumps(state, ensure_ascii=False)}\\n\\n"

                await asyncio.sleep(0.25)
        except asyncio.CancelledError:
            return

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


@app.on_event("shutdown")
def on_shutdown() -> None:
    _stop_live_worker()
