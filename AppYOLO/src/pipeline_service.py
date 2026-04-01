from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from src.inference.infer import YOLOInfer


def find_escape_path(start: str, dangers: dict[str, bool], adj: dict[str, list[str]], safe_targets: list[str]) -> Optional[list[str]]:
    if not safe_targets:
        return None

    visited = set()
    queue = deque([(start, [start])])

    while queue:
        curr, path = queue.popleft()
        if curr in visited:
            continue
        visited.add(curr)

        if curr in safe_targets:
            return path

        for neigh in adj.get(curr, []):
            if neigh not in visited and not dangers.get(neigh, True):
                queue.append((neigh, path + [neigh]))

    return None


def run_main_pipeline(
    infer: YOLOInfer,
    image_path: str,
    video_path: str,
    output_video_path: str,
    with_decision: bool = True,
    display: bool = False,
) -> dict[str, Any]:
    frame_id = 1042
    yolo_results, decision, vision_temp = infer.run_with_decision(
        image_path=image_path,
        save=True,
        frame_id=frame_id,
    )

    video_results = infer.run(
        video_path,
        save_path=output_video_path,
        with_decision=with_decision,
        display=display,
    )

    return {
        "image": {
            "image_path": image_path,
            "frame_id": frame_id,
            "detections": yolo_results,
            "vision_temperature_celsius": vision_temp,
            "decision": decision,
        },
        "video": {
            "video_path": video_path,
            "output_video_path": output_video_path,
            "frame_count": len(video_results),
            "sample": video_results[0] if video_results else {},
        },
    }


def run_vcn_pipeline(
    infer: YOLOInfer,
    package_dir: str = "WebCamPackage",
    map_json_path: str = "WebCamPackage/_map.json",
) -> dict[str, Any]:
    if not os.path.exists(map_json_path):
        raise FileNotFoundError(f"Map json not found: {map_json_path}")

    with open(map_json_path, "r", encoding="utf-8") as fp:
        map_data = json.load(fp)

    results: dict[str, Any] = {}
    file_list = list(map_data.keys())

    for idx, filename in enumerate(file_list):
        image_path = os.path.join(package_dir, filename)
        if not os.path.exists(image_path):
            continue

        yolo_results, decision, vision_temp = infer.run_with_decision(
            image_path=image_path,
            save=True,
            frame_id=2000 + idx,
        )

        suggested_action = decision.get("decision", {}).get("suggested_action", "CONTINUE_MONITORING")
        is_danger = suggested_action in ["EVACUATE_AND_SHUTDOWN", "ALERT", "EVACUATE"]

        label_map = {
            "left1.png": "left1",
            "mid.jpg": "mid",
            "right1.png": "right1",
            "right2.jpg": "right2",
        }
        label = label_map.get(filename, filename.split(".")[0])

        results[filename] = {
            "is_danger": is_danger,
            "vision_temp": vision_temp,
            "decision": decision,
            "suggested_action": suggested_action,
            "coord": [int(c.strip()) for c in map_data[filename].split(",")],
            "label": label,
            "annotated_path": f"outputs/images/{filename}",
            "yolo_results": yolo_results,
        }

    if not results:
        raise RuntimeError("No package image was processed.")

    sample_data = next(iter(results.values()))
    sample_path = sample_data["annotated_path"]
    sample_img = cv2.imread(sample_path)
    if sample_img is None:
        first_file = list(results.keys())[0]
        sample_img = cv2.imread(os.path.join(package_dir, first_file))

    if sample_img is None:
        raise RuntimeError("Unable to read sample image for composing map.")

    orig_h, orig_w = sample_img.shape[:2]

    map_size = 1000
    monitor_width = 250
    monitors_per_row = 4

    monitor_h = int(monitor_width * orig_h / orig_w)

    map_w = map_size
    map_h = map_size // 2
    map_img = np.ones((map_h, map_w, 3), dtype=np.uint8) * 255

    scale_x = map_w / 1000.0
    scale_y = map_h / 1000.0

    adj = {
        "left1": ["mid"],
        "mid": ["left1", "right1", "right2"],
        "right1": ["mid"],
        "right2": ["mid"],
    }

    connection_pairs = [("left1", "mid"), ("mid", "right1"), ("mid", "right2")]
    for a, b in connection_pairs:
        ap = None
        bp = None
        for data in results.values():
            if data["label"] == a:
                ax, ay = data["coord"]
                ap = (int(ax * scale_x), int(ay * scale_y))
            if data["label"] == b:
                bx, by = data["coord"]
                bp = (int(bx * scale_x), int(by * scale_y))

        if ap and bp:
            cv2.line(map_img, ap, bp, (180, 180, 180), 6)

    for data in results.values():
        x, y = data["coord"]
        px = int(x * scale_x)
        py = int(y * scale_y)
        color = (0, 0, 255) if data["is_danger"] else (0, 255, 0)
        cv2.circle(map_img, (px, py), 22, color, -1)
        cv2.circle(map_img, (px, py), 27, (0, 0, 0), 4)
        cv2.putText(
            map_img,
            data["label"],
            (px + 35, py + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    dangers = {data["label"]: data["is_danger"] for data in results.values()}
    escape_texts = {}
    exit_nodes = ["right2"]

    for start_label in ["left1", "mid", "right1", "right2"]:
        if start_label not in dangers or not dangers[start_label]:
            escape_texts[start_label] = f"{start_label} The location is currently safe."
            continue

        safe_exits = [n for n in exit_nodes if n in dangers and not dangers[n]]
        if not safe_exits:
            safe_exits = [n for n in adj if n in dangers and not dangers[n]]

        if not safe_exits:
            escape_texts[start_label] = f"{start_label} There is a fire, but no safe route. Please take shelter."
            continue

        path = find_escape_path(start_label, dangers, adj, safe_exits)
        if path and len(path) > 1:
            route = " -> ".join(path[1:])
            escape_texts[start_label] = f"{start_label} on fire, please to {route}."
        else:
            escape_texts[start_label] = f"{start_label} Fire has broken out, please shelter in place."

    num_monitor_rows = (len(file_list) + monitors_per_row - 1) // monitors_per_row
    total_monitor_height = num_monitor_rows * monitor_h

    canvas_w = map_size
    canvas_h = map_h + total_monitor_height + 520
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    canvas[0:map_h, 0:map_w] = map_img

    order = ["left1.png", "mid.jpg", "right1.png", "right2.jpg"]
    monitor_offset_x = 0
    grid_y = map_h

    for idx, fn in enumerate(order):
        row = idx // monitors_per_row
        col = idx % monitors_per_row

        if fn in results:
            img = cv2.imread(results[fn]["annotated_path"])
            if img is None:
                img = np.ones((orig_h, orig_w, 3), dtype=np.uint8) * 100
            img = cv2.resize(img, (monitor_width, monitor_h))

            label = results[fn]["label"]
            status_str = "[Danger]" if results[fn]["is_danger"] else "[Safety]"
            cv2.putText(img, f"{label} {status_str}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 4)
            cv2.putText(img, f"{label} {status_str}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
        else:
            img = np.ones((monitor_h, monitor_width, 3), dtype=np.uint8) * 180

        x = monitor_offset_x + col * monitor_width
        y = grid_y + row * monitor_h
        canvas[y:y + monitor_h, x:x + monitor_width] = img

    text_y = map_h + total_monitor_height + 40
    cv2.putText(canvas, "Escape Instructions at each location", (80, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (0, 0, 0), 3)

    y_offset = text_y + 70
    for label in ["left1", "mid", "right1", "right2"]:
        if label in escape_texts:
            txt = escape_texts[label]
            cv2.putText(canvas, f"{label} : {txt}", (80, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 2, cv2.LINE_AA)
            y_offset += 68

    Path("outputs").mkdir(parents=True, exist_ok=True)
    output_path = "outputs/b1_composite_map.png"
    cv2.imwrite(output_path, canvas)

    return {
        "processed_files": list(results.keys()),
        "output_path": output_path,
        "escape_texts": escape_texts,
        "camera_results": results,
    }
