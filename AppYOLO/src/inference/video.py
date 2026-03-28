# src/inference/video.py（完整替換）
import cv2
import time
import numpy as np
from datetime import datetime, timezone
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
from .utils import convert_to_yolo_format,FireTemperatureEstimator
from ..decision_engine import SafetyDecisionEngine
 
 



class VideoInfer:
    def __init__(self, model_path="models/release.pt", tracker="botsort.yaml"):
        self.model = YOLO(model_path)
        self.tracker = tracker
        self.decision_engine = SafetyDecisionEngine(fps=30, alarm_threshold=0.75)

        
        # --- 新增：建立測溫器實體 ---
        self.temp_estimator = FireTemperatureEstimator()
   
    def _to_yolo_format_str(self, result):
        """產生 decision_engine 需要的 YOLO txt 字串"""
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

    def _create_professional_dashboard(self, decision_result, vision_temp, video_height):
        """專業儀表板（右側獨立面板）—— 乾淨、美觀、不會文字亂碼
        已完全適配影片高度：所有位置、字體、進度條、間距均按比例縮放，
        確保任何解析度（含小影片）都不會破圖或文字溢出。
        """
        if video_height <= 0:
            # 極端保護
            panel = np.zeros((100, 440, 3), dtype=np.uint8)
            panel[:] = (28, 28, 32)
            cv2.putText(panel, "SYSTEM ERROR", (80, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
            return panel

        # === 比例縮放核心（以 720p 為設計基準）===
        BASE_H = 720.0
        scale = video_height / BASE_H
        panel_w = int(440 * scale)

        panel = np.zeros((video_height, panel_w, 3), dtype=np.uint8)
        panel[:] = (28, 28, 32)  # 深色現代背景

        # 快捷縮放函數
        def sx(x: float) -> int:
            return int(x * scale)

        def sy(y: float) -> int:
            return int(y * scale)

        def sfs(fs: float) -> float:
            return fs * scale

        def sth(th: int) -> int:
            return max(1, int(th * scale))

        if decision_result["status"] != "success":
            cv2.putText(panel, "SYSTEM ERROR", (sx(80), sy(200)),
                        cv2.FONT_HERSHEY_DUPLEX, sfs(1.0), (0, 0, 255), sth(2))
            return panel

        alarm = decision_result["decision"]["trigger_alarm"]
        panel_color = (0, 0, 255) if alarm else (0, 220, 0)  # 紅 / 綠
        risk = decision_result["decision"]["risk_score"]
        action = decision_result["decision"]["suggested_action"]
        trace = decision_result["explainability"]["trace_message"]

        # === Header ===
        cv2.rectangle(panel, (sx(20), sy(20)), (panel_w - sx(20), sy(75)), panel_color, -1)
        cv2.putText(panel, "FIRE SAFETY MONITOR", (sx(45), sy(55)),
                    cv2.FONT_HERSHEY_DUPLEX, sfs(0.95), (255, 255, 255), sth(2))

        # === Vision Temperature ===
        cv2.rectangle(panel, (sx(30), sy(95)), (panel_w - sx(30), sy(165)), (45, 45, 50), -1)
        cv2.putText(panel, "VISION TEMPERATURE", (sx(50), sy(118)),
                    cv2.FONT_HERSHEY_SIMPLEX, sfs(0.58), (180, 180, 180), sth(1))
        temp_str = f"{vision_temp:.0f}°C" if vision_temp is not None else "N/A"
        cv2.putText(panel, temp_str, (sx(68), sy(155)),
                    cv2.FONT_HERSHEY_DUPLEX, sfs(1.35), panel_color, sth(3))

        # === Risk Score + 進度條 ===
        cv2.rectangle(panel, (sx(30), sy(185)), (panel_w - sx(30), sy(245)), (45, 45, 50), -1)
        cv2.putText(panel, "RISK SCORE", (sx(50), sy(208)),
                    cv2.FONT_HERSHEY_SIMPLEX, sfs(0.58), (180, 180, 180), sth(1))
        cv2.putText(panel, f"{risk:.2f}", (sx(70), sy(240)),
                    cv2.FONT_HERSHEY_DUPLEX, sfs(1.25), panel_color, sth(3))

        # 風險進度條（寬度也隨比例縮放）
        bar_max = int(340 * scale)
        bar_length = int(risk * bar_max)
        bar_y = sy(255)
        bar_bottom = sy(265)
        cv2.rectangle(panel, (sx(50), bar_y), (sx(50) + bar_length, bar_bottom), panel_color, -1)
        cv2.rectangle(panel, (sx(50), bar_y), (sx(50) + bar_max, bar_bottom), (90, 90, 95), sth(2))

        # === Decision Action ===
        cv2.rectangle(panel, (sx(30), sy(285)), (panel_w - sx(30), sy(345)), (45, 45, 50), -1)
        cv2.putText(panel, "DECISION", (sx(50), sy(308)),
                    cv2.FONT_HERSHEY_SIMPLEX, sfs(0.58), (180, 180, 180), sth(1))
        cv2.putText(panel, action, (sx(70), sy(335)),
                    cv2.FONT_HERSHEY_DUPLEX, sfs(0.95), panel_color, sth(2))

        # === Status Message（避免亂碼，使用英文關鍵字 + 原 trace 前半部）===
        cv2.rectangle(panel, (sx(30), sy(365)), (panel_w - sx(30), video_height - sx(30)), (45, 45, 50), -1)
        cv2.putText(panel, "CURRENT STATUS", (sx(50), sy(388)),
                    cv2.FONT_HERSHEY_SIMPLEX, sfs(0.58), (180, 180, 180), sth(1))

        status_text = trace[:110] if len(trace) > 110 else trace
        lines = [status_text[i:i + 38] for i in range(0, len(status_text), 38)]
        start_y = sy(415)
        line_spacing = sy(23)
        for i, line in enumerate(lines[:5]):
            y_pos = start_y + i * line_spacing
            cv2.putText(panel, line, (sx(50), y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, sfs(0.62), (230, 230, 230), sth(1))

        return panel

    def run(
        self,
        video_path,
        save_path=None,
        conf=0.25,
        iou=0.7,
        imgsz=640,
        half=False,
        device=None,
        with_decision: bool = False,  # 開啟專業儀表板
    ):
        cap = cv2.VideoCapture(video_path)
        all_results = []
        writer = None

        # === 預先取得影片尺寸，決定輸出解析度（含儀表板）===
        if save_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 當啟用儀表板時，輸出寬度 = 原寬 + 自適應儀表板寬（避免破圖）
            panel_w = 0
            if with_decision:
                scale = h / 720.0
                panel_w = int(440 * scale)
            out_w = w + panel_w
            out_h = h

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (out_w, out_h))

        frame_id = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO 追蹤
            results = self.model.track(
                frame, persist=True, tracker=self.tracker,
                verbose=False, conf=conf, iou=iou,
                imgsz=imgsz, half=half, device=device,
            )
            result = results[0]
            yolo_result = convert_to_yolo_format(result)
            all_results.append({"frame_id": frame_id, "detections": yolo_result})

            # 乾淨的 annotated 畫面（僅 bounding box + Track ID）
            annotated = result.plot(conf=False, labels=False)
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls.item())
                track_id = int(box.id.item()) if box.id is not None else -1
                class_name = result.names[cls_id]
                color = colors(cls_id, True)
                label = class_name
                if track_id >= 0:
                    label += f" : {track_id}"
                cv2.putText(annotated, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # ==================== 專業儀表板 GUI ====================
            frame_to_write = annotated
            if with_decision:
                vision_temp = self.temp_estimator._estimate_temperature_from_frame(frame, result)
                timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                visual_objects_str = self._to_yolo_format_str(result)
                payload = {
                    "context": {"timestamp": timestamp, "frame_id": frame_id},
                    "perceptions": {
                        "visual_objects": visual_objects_str,
                        "environmental_sensors": {
                            "temperature_celsius": vision_temp if vision_temp is not None else 25.4
                        }
                    }
                }
                decision_result = self.decision_engine.evaluate_payload(payload)

                # 建立右側專業儀表板（已自適應影片尺寸）
                dashboard = self._create_professional_dashboard(
                    decision_result, vision_temp, annotated.shape[0]
                )
                # 左右合併成一個漂亮的監控畫面
                combined = np.hstack((annotated, dashboard))

                # 即時顯示
                cv2.imshow("🔥 YOLO Fire Safety - Professional Monitor", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # 同時把決策資訊存入回傳結果
                all_results[-1]["vision_temp"] = vision_temp
                all_results[-1]["decision"] = decision_result

                frame_to_write = combined  # ← 改成包含儀表板的畫面

            # 寫入影片（現在會同時存入 UI 加工後的畫面）
            if writer:
                writer.write(frame_to_write)

            frame_id += 1

        cap.release()
        if writer:
            writer.release()
        if with_decision:
            cv2.destroyAllWindows()

        # FPS 統計
        total_time = time.time() - start_time
        fps = frame_id / total_time if frame_id > 0 else 0
        print(f"\nProcess Complete！")
        print(f" Total Frame : {frame_id} frame")
        print(f" Time Cost : {total_time:.2f} second")
        print(f" Average FPS : {fps:.2f}")
        return all_results