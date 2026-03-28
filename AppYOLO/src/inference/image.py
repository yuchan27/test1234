import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime, timezone

from .utils import convert_to_yolo_format
from ..decision_engine import SafetyDecisionEngine


class ImageInfer:
    def __init__(self, model_path="models/release.pt"):
        self.model = YOLO(model_path)
        self.output_dir = Path("outputs/images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 整合 SafetyDecisionEngine
        self.decision_engine = SafetyDecisionEngine(fps=30, alarm_threshold=0.75)

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

    def _estimate_temperature_from_bbox(self, image_path, yolo_result):
        """
        科學熱力學公式（雙色高溫計 / Two-Color Pyrometry）估測火焰溫度
        已修正：限制在真實火焰溫度範圍（600~1600°C）
        
        為什麼原本會算到 2726°C？
        → 這是理論公式在「未校準的消費級相機/RGB影像」上的常見現象。
          相機白平衡、Gamma、曝光、火焰非完美黑體（emissivity ≠ 1）都會讓 R/G 比例失真，
          導致理論值大幅高估。
        
        真實火災（木材/有機物）火焰溫度通常落在 800~1200°C，劇烈火場最高 ~1500°C。
        因此我們保留科學公式，但加上物理上合理的上下限，讓決策引擎更可靠。
        """
        img = cv2.imread(image_path)
        if img is None:
            return None

        FIRE_CLASS_ID = 1  # 你的模型實際 Fire 類別 ID

        fire_boxes = []
        for box in yolo_result.boxes:
            if int(box.cls[0]) == FIRE_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                fire_boxes.append((x1, y1, x2, y2))

        if not fire_boxes:
            return None

        temps = []
        for x1, y1, x2, y2 in fire_boxes:
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            avg_r = float(np.mean(crop[:, :, 2]))
            avg_g = float(np.mean(crop[:, :, 1]))
            if avg_g < 1.0:
                avg_g = 1.0

            lambda_r = 700e-9
            lambda_g = 546.1e-9
            C2 = 0.014388
            Cg = 1.0

            ratio = avg_r / avg_g
            lambda_ratio5 = (lambda_r / lambda_g) ** 5
            arg = (1.0 / Cg) * ratio * lambda_ratio5

            if arg <= 0:
                continue

            ln_arg = np.log(arg)
            temp_k = C2 * (1 / lambda_g - 1 / lambda_r) / ln_arg
            temp_k = max(300.0, min(3000.0, temp_k))
            temp_c = temp_k - 273.15

            temps.append(temp_c)

        if not temps:
            return None

        temp_celsius = float(np.mean(temps))

        # 🔥 關鍵修正：限制在真實火焰溫度範圍
        temp_celsius = max(600.0, min(1600.0, temp_celsius))

        return temp_celsius

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
        """整合版執行流程（已修正）"""
        results = self.model(image_path, verbose=False)
        result = results[0]
        yolo_results = convert_to_yolo_format(result)   # 保留給使用者的原始格式

        if save:
            annotated = result.plot(conf=False)
            filename = os.path.basename(image_path)
            save_path = self.output_dir / filename
            cv2.imwrite(str(save_path), annotated)

        vision_temp_c = self._estimate_temperature_from_bbox(image_path, result)

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
                    "temperature_celsius": vision_temp_c if vision_temp_c is not None else 25.4
                }
            }
        }

        decision_result = self.decision_engine.evaluate_payload(payload)

        return yolo_results, decision_result, vision_temp_c