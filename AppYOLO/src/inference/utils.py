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
        
        # 預先計算波長比例常數，提升效能
        self.lambda_ratio5 = (self.lambda_r / self.lambda_g) ** 5
        self.wavelength_term = self.C2 * (1 / self.lambda_g - 1 / self.lambda_r)

    def _estimate_temperature_from_frame(self, frame, yolo_result):
        if frame is None:
            return None
            
        FIRE_CLASS_ID = 1 # 確保這是你的「火」類別 ID
        # SMOKE_CLASS_ID = 0 # 如果你有煙霧類別，煙霧無法用RGB測溫，直接返回 ambient_temp

        fire_boxes = []
        for box in yolo_result.boxes:
            if int(box.cls[0]) == FIRE_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                fire_boxes.append((x1, y1, x2, y2))
                
        if not fire_boxes:
            return self.ambient_temp # 沒火時回傳常溫

        temps = []
        for x1, y1, x2, y2 in fire_boxes:
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
            # 條件: 亮度足夠 (R>100), 符合火焰色彩特性 (R>G>B), 排除過曝 (R<250, G<250)
            valid_mask = (R > 100) & (R > G) & (G > B) & (R < 250) & (G < 250) & (G > 10)
            
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
            
            # 物理極限濾波：火焰溫度合理範圍大約在 400°C ~ 1500°C 之間
            pixel_temps_c = pixel_temps_c[(pixel_temps_c >= 300) & (pixel_temps_c <= 2000)]

            if len(pixel_temps_c) > 0:
                # 4. 取中位數而非平均，避免被極端值干擾
                box_temp = np.median(pixel_temps_c)
                temps.append(box_temp)

        if not temps:
            return self.ambient_temp # 如果有框但算不出溫度，回傳預設常溫

        # 取所有火焰框的最大溫度或平均溫度作為影像代表
        final_temp = float(np.max(temps)) 
        return round(final_temp, 2)
