# src/inference/infer.py
import os
from pathlib import Path

from .image import ImageInfer
from .video import VideoInfer


class YOLOInfer:
    """
    統一入口類別（Facade Pattern）
    負責根據檔案類型自動分派給 ImageInfer 或 VideoInfer
    """
    def __init__(self, model_path="models/release.pt"):
        self.image_infer = ImageInfer(model_path)   # 已內建 SafetyDecisionEngine + bbox 科學溫度估測
        self.video_infer = VideoInfer(model_path)

    def run(
        self,
        path: str,
        save_path: str | None = None,
        with_decision: bool = False,
        display: bool = True,
    ):
            if path.lower().endswith((".jpg", ".png", ".jpeg")):
                return self.image_infer.run(path, save=True)

            elif path.lower().endswith((".mp4", ".avi", ".mov")):
                return self.video_infer.run(
                    path,
                    save_path,
                    with_decision=with_decision,
                    display=display,
                )

            else:
                raise ValueError(f"Unsupported file format: {path}")

    def run_with_decision(self, image_path: str, save: bool = True, frame_id: int = 0):
        """
        【新功能】整合版執行流程（僅支援單張影像）
        1. YOLO 推論
        2. 透過 bounding box 科學熱力學公式（雙色高溫計 + Planck 定律）計算 vision temperature_celsius
        3. 自動建構 Payload 並呼叫 SafetyDecisionEngine
        4. 回傳三元組：(yolo_results, decision_result, vision_temp_celsius)

        影片暫不支援（未來可擴充為 frame-by-frame 決策）
        """
        if not image_path.lower().endswith((".jpg", ".png", ".jpeg")):
            raise ValueError(
                "run_with_decision 目前僅支援單張影像檔案 (.jpg/.png/.jpeg)。\n"
                "影片請使用原 run() 方法，或未來擴充 frame-by-frame 決策。"
            )

        # 直接委派給 ImageInfer（已完整整合溫度計算與決策引擎）
        # 注意：這裡把 save 參數正確傳遞下去（原本你寫成 save: str | None 是筆誤）
        return self.image_infer.run_with_decision(
            image_path=image_path,
            save=save,           # ← 正確傳遞 bool
            frame_id=frame_id
        )


# ==================== 正確的使用範例（已修正） ====================
if __name__ == "__main__":
    infer = YOLOInfer(model_path="models/release.pt")

    # ───── 圖片（整合決策模式） ─────
    # 請把下面的路徑換成你電腦上「真實存在的」火災/煙霧圖片
    real_image_path = "test/fire_test.jpg"          # ←←← 改成你的實際檔案路徑

    yolo_results, decision, vision_temp = infer.run_with_decision(
        image_path=real_image_path,
        save=True,      # 是否儲存 annotated 圖片
        frame_id=1042
    )

    print("\n=== 圖片整合決策結果 ===")
    print("YOLO 結果:", yolo_results)
    print("視覺估測火焰溫度:", vision_temp, "°C" if vision_temp is not None else "（無火）")
    print("決策:", decision["decision"]["suggested_action"])
    print("解釋:", decision["explainability"]["trace_message"])

    # ───── 影片（原始模式） ─────
    result = infer.run("forest1.avi", save_path="outputs/out.mp4")
    print("\n=== 影片處理完成 ===")
    print("影片輸出結果:", result[0] if result else "None")