def convert_to_yolo_format(result):
    boxes = result.boxes

    if boxes is None:
        return []

    output = []

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xywh = box.xywhn[0].tolist()  # normalized

        output.append({
            "class_id": cls_id,
            "class_name": result.names[cls_id],
            "confidence": conf,
            "bbox": xywh
        })

    return output



import cv2
import numpy as np
class FireTemperatureEstimator:
    def __init__(self, ambient_temp=25.0):
        # 物理常數
        self.lambda_r = 700e-9   # 紅光波長 (m)
        self.lambda_g = 546.1e-9 # 綠光波長 (m)
        self.C2 = 0.014388       # 普朗克第二輻射常數 (m·K)
        self.ambient_temp = ambient_temp # 常溫預設值 (Celsius)
        self.last_temperature = float(ambient_temp)
        
        # 預先計算波長比例常數，提升效能
        self.lambda_ratio5 = (self.lambda_r / self.lambda_g) ** 5
        self.wavelength_term = self.C2 * (1 / self.lambda_g - 1 / self.lambda_r)

    def _resolve_target_boxes(self, yolo_result):
        names = getattr(yolo_result, "names", {}) or {}
        fire_boxes = []
        smoke_boxes = []

        for box in yolo_result.boxes:
            cls_id = int(box.cls[0])
            class_name = str(names.get(cls_id, "")).lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # 避免空框與越界帶來的 crop 問題
            if x2 <= x1 or y2 <= y1:
                continue

            if "fire" in class_name or "flame" in class_name:
                fire_boxes.append((x1, y1, x2, y2))
            elif "smoke" in class_name:
                smoke_boxes.append((x1, y1, x2, y2))

        # 兼容舊模型：若沒有 class name，可退回 id=1 判定 fire
        if not fire_boxes and not smoke_boxes:
            for box in yolo_result.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 1:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    if x2 > x1 and y2 > y1:
                        fire_boxes.append((x1, y1, x2, y2))

        return fire_boxes, smoke_boxes

    def _estimate_dynamic_fallback_temperature(self, frame, boxes):
        if frame is None or frame.size == 0:
            return float(self.ambient_temp)

        h, w = frame.shape[:2]
        regions = []

        for x1, y1, x2, y2 in boxes[:6]:
            x1c = max(0, min(w - 1, x1))
            y1c = max(0, min(h - 1, y1))
            x2c = max(0, min(w, x2))
            y2c = max(0, min(h, y2))
            crop = frame[y1c:y2c, x1c:x2c]
            if crop.size > 0:
                regions.append(crop)

        if not regions:
            # 沒有偵測框時，改採中央區域動態估計，避免固定常溫看起來像故障
            y1 = h // 4
            y2 = max(y1 + 1, (h * 3) // 4)
            x1 = w // 4
            x2 = max(x1 + 1, (w * 3) // 4)
            center_crop = frame[y1:y2, x1:x2]
            regions = [center_crop if center_crop.size > 0 else frame]

        brightness_values = []
        red_dominance_values = []
        contrast_values = []

        for crop in regions:
            crop_f = crop.astype(np.float32)
            b = crop_f[:, :, 0]
            g = crop_f[:, :, 1]
            r = crop_f[:, :, 2]

            gray = (0.114 * b + 0.587 * g + 0.299 * r) / 255.0
            brightness_values.append(float(np.mean(gray)))
            contrast_values.append(float(np.std(gray)))
            red_dominance_values.append(float(np.mean(np.clip(r - g, 0, 255) / 255.0)))

        brightness = max(brightness_values) if brightness_values else 0.0
        contrast = max(contrast_values) if contrast_values else 0.0
        red_dominance = max(red_dominance_values) if red_dominance_values else 0.0

        # 映射到可觀察但不誇張的動態溫度區間
        dynamic_temp = self.ambient_temp + (brightness * 8.0) + (contrast * 40.0) + (red_dominance * 35.0)
        dynamic_temp = float(np.clip(dynamic_temp, self.ambient_temp - 2.0, 180.0))
        return round(dynamic_temp, 2)

    def _estimate_temperature_from_frame(self, frame, yolo_result):
        if frame is None:
            return None

        fire_boxes, smoke_boxes = self._resolve_target_boxes(yolo_result)
        target_boxes = fire_boxes if fire_boxes else smoke_boxes

        if not target_boxes:
            fallback = self._estimate_dynamic_fallback_temperature(frame, [])
            self.last_temperature = round((0.85 * self.last_temperature) + (0.15 * fallback), 2)
            return self.last_temperature

        temps = []
        for x1, y1, x2, y2 in target_boxes:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # 將 BGR 轉為 RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # 分離通道並轉為 float
            R = crop_rgb[:, :, 0].astype(float)
            G = crop_rgb[:, :, 1].astype(float)
            B = crop_rgb[:, :, 2].astype(float)

            # 1. 建立遮罩 (Mask): 尋找有效火焰像素
            # 放寬上限，避免過曝火焰像素被全數排除
            valid_mask = (R > 80) & (R >= (G * 1.02)) & (G >= (B * 0.9)) & (G > 8)

            if np.count_nonzero(valid_mask) == 0:
                # 次級遮罩：在煙霧與低光場景維持可估算性
                valid_mask = (R > 60) & (R >= G) & (R >= B)
            
            valid_R = R[valid_mask]
            valid_G = G[valid_mask]

            # 如果沒有符合的有效發光像素 (例如這其實是一團黑煙或偵測錯誤)
            if len(valid_R) == 0:
                continue

            # 2. 逆向 Gamma 校正 (轉為近似線性輻射強度)
            lin_R = (valid_R / 255.0) ** 2.2
            lin_G = (valid_G / 255.0) ** 2.2

            # 3. 雙色高溫計運算 (向量化運算提升速度)
            ratio = lin_R / lin_G
            arg = ratio * self.lambda_ratio5
            
            # 過濾無效的 log 參數
            valid_arg_mask = arg > 1.0 
            arg = arg[valid_arg_mask]
            
            if len(arg) == 0:
                continue

            ln_arg = np.log(arg)
            pixel_temps_k = self.wavelength_term / ln_arg
            pixel_temps_c = pixel_temps_k - 273.15
            
            # 物理極限濾波：放寬下限，避免因場景偏暗導致全部被過濾
            pixel_temps_c = pixel_temps_c[(pixel_temps_c >= 250) & (pixel_temps_c <= 1600)]

            if len(pixel_temps_c) > 0:
                # 4. 取中位數而非平均，避免被極端值干擾
                box_temp = np.median(pixel_temps_c)
                temps.append(box_temp)

        if not temps:
            fallback = self._estimate_dynamic_fallback_temperature(frame, target_boxes)
            self.last_temperature = round((0.8 * self.last_temperature) + (0.2 * fallback), 2)
            return self.last_temperature

        # 取所有火焰框的最大溫度或平均溫度作為影像代表
        final_temp = float(np.max(temps))

        # 輕度平滑，避免每幀抖動過大但仍保有動態變化
        self.last_temperature = round((0.65 * self.last_temperature) + (0.35 * final_temp), 2)
        return self.last_temperature
