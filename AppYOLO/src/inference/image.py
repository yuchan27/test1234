import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime, timezone

from .utils import convert_to_yolo_format,FireTemperatureEstimator
from ..decision_engine import SafetyDecisionEngine








class ImageInfer:
    def __init__(self, model_path="models/release.pt"):
        self.model = YOLO(model_path)
        self.output_dir = Path("outputs/images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 整合 SafetyDecisionEngine
        self.decision_engine = SafetyDecisionEngine(fps=30, alarm_threshold=0.55)
        self.temp_estimator = FireTemperatureEstimator()

    def run(self, image_path, save=True):
        """原有功能（完全不變）"""
        results = self.model(image_path, verbose=False)
        result = results[0]

        yolo_results = convert_to_yolo_format(result)

        if save:
            annotated = result.plot(conf=False)
            filename = os.path.basename(image_path)
            save_path = self.output_dir / filename
            cv2.imwrite(str(save_path), annotated)

        return yolo_results

     

    def _to_yolo_format_str(self, result):
        """強制轉成 decision_engine 相容的 YOLO txt 字串（含 confidence）"""
        if len(result.boxes) == 0:
            return ""

        cls_list = result.boxes.cls.tolist()
        conf_list = result.boxes.conf.tolist()
        xywhn_list = result.boxes.xywhn.tolist()

        lines = []
        for cls_id, conf_val, (x, y, w, h) in zip(cls_list, conf_list, xywhn_list):
            line = f"{int(cls_id)} {x:.4f} {y:.4f} {w:.4f} {h:.4f} {conf_val:.4f}"
            lines.append(line)
        return "\n".join(lines)

    def run_with_decision(self, image_path, save=True, frame_id=0):
        """整合版執行流程（修復參數錯誤）"""
        results = self.model(image_path, verbose=False)
        result = results[0]
        yolo_results = convert_to_yolo_format(result)

        # 1. 取得影像矩陣：使用 result.orig_img (YOLO 內建讀取的 BGR 影像)
        # 這樣就不用再跑一次 cv2.imread(image_path) 浪費效能
        img_frame = result.orig_img

        if save:
            annotated = result.plot(conf=False)
            filename = os.path.basename(image_path)
            save_path = self.output_dir / filename
            cv2.imwrite(str(save_path), annotated)

        # 2. 這裡修正：傳入 img_frame 而不是 image_path
        vision_temp_c = self.temp_estimator._estimate_temperature_from_frame(img_frame, result)

        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        visual_objects_str = self._to_yolo_format_str(result)

        payload = {
            "context": {
                "timestamp": timestamp,
                "frame_id": frame_id
            },
            "perceptions": {
                "visual_objects": visual_objects_str,
                "environmental_sensors": {
                    # 確保如果 vision_temp_c 是 None (沒火)，給予預設值
                    "temperature_celsius": vision_temp_c if vision_temp_c is not None else 25.4
                }
            }
        }

        decision_result = self.decision_engine.evaluate_payload(payload)

        return yolo_results, decision_result, vision_temp_c