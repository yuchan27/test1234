import cv2
import time   
from ultralytics import YOLO
from ultralytics.utils.plotting import colors   # ← 新增這行（取得 bounding box 同色）
from .utils import convert_to_yolo_format


class VideoInfer:
    def __init__(self, model_path="models/release.pt", tracker="botsort.yaml"):
        """
        tracker 可選：
        - "botsort.yaml"  → SOTA 準度最高（推薦）
        - "bytetrack.yaml" → 速度最快
        """
        self.model = YOLO(model_path)
        self.tracker = tracker

    def run(
        self,
        video_path,
        save_path=None,
        conf=0.25,
        iou=0.7,
        imgsz=640,
        half=False,
        device=None,
    ):
        cap = cv2.VideoCapture(video_path)

        all_results = []

        writer = None
        if save_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

        frame_id = 0
        start_time = time.time()   # 開始計時（用來算 FPS）

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # SOTA 追蹤（BoT-SORT / ByteTrack）
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker,
                verbose=False,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                half=half,
                device=device,
            )
            result = results[0]

            # 取得 YOLO 格式結果
            yolo_result = convert_to_yolo_format(result)

            all_results.append({
                "frame_id": frame_id,
                "detections": yolo_result
            })

            # === 重點修改：只顯示類別名稱 + Track ID，不顯示 class id、無文字黑框、文字顏色與 bounding box 相同 ===
            # 先只畫 bounding box + track ID（不畫預設的 class name）
            annotated = result.plot(conf=False, labels=False)

            # 手動在框上方畫「類別名稱」和「Track ID」
            boxes = result.boxes
            for box in boxes:
                # 取得座標、class id、track id、類別名稱
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls.item())
                track_id = int(box.id.item()) if box.id is not None else -1
                class_name = result.names[cls_id]                     # ← 只取類別名稱

                # 取得與 bounding box 完全相同的顏色（BGR）
                color = colors(cls_id, True)

                # 顯示文字：類別名稱 + Track ID（例如：person Track:12）
                label = class_name
                if track_id >= 0:
                    label += f" : {track_id}"

                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x, text_y = int(x1), int(y1) - 10

                # ← 完全移除黑框（不要 cv2.rectangle）

                # 直接畫文字，顏色與 bounding box 一致
                cv2.putText(annotated, label, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 寫入影片
            if writer:
                writer.write(annotated)

            frame_id += 1

        cap.release()
        if writer:
            writer.release()

        # === 重點修改 2：結束時印出 FPS ===
        total_time = time.time() - start_time
        fps = frame_id / total_time if frame_id > 0 else 0
        print(f"\nProcess Complete！")
        print(f"   Total Frame : {frame_id} frame")
        print(f"   Time Cost : {total_time:.2f} second")
        print(f"   Average FPS : {fps:.2f}")

        return all_results