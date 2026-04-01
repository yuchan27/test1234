# main.py
 

import os
import json
import cv2
import numpy as np
from collections import deque

from src.inference.infer import YOLOInfer


def find_escape_path(start: str, dangers: dict, adj: dict, safe_targets: list) -> list | None:
    """BFS 尋找最短安全逃生路徑（只走安全點）"""
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


if __name__ == "__main__":
    print("Campus Basement B1 Four-Camera Integrated Decision Simulation")
    
    package_dir = "WebCamPackage"
    map_json_path = os.path.join(package_dir, "_map.json")
    
    if not os.path.exists(map_json_path):
        raise FileNotFoundError(f"找不到地圖設定檔：{map_json_path}")
    
    with open(map_json_path, "r", encoding="utf-8") as f:
        map_data = json.load(f)
    
    infer = YOLOInfer("models/release11.pt")
    
    results = {}
    file_list = list(map_data.keys())
    
    for idx, filename in enumerate(file_list):
        image_path = os.path.join(package_dir, filename)
        if not os.path.exists(image_path):
            print(f"⚠️  跳過不存在檔案：{filename}")
            continue
            
        print(f"正在處理 {filename} ...")
        
        yolo_results, decision, vision_temp = infer.run_with_decision(
            image_path=image_path,
            save=True,
            frame_id=2000 + idx
        )
        
        suggested_action = decision.get("decision", {}).get("suggested_action", "CONTINUE_MONITORING")
        is_danger = suggested_action in ["EVACUATE_AND_SHUTDOWN", "ALERT", "EVACUATE"]
        
        label_map = {
            "left1.png": "left1",
            "mid.jpg": "mid",
            "right1.png": "right1",
            "right2.jpg": "right2"
        }
        label = label_map.get(filename, filename.split('.')[0])
        
        results[filename] = {
            "is_danger": is_danger,
            "vision_temp": vision_temp,
            "decision": decision,
            "suggested_action": suggested_action,
            "coord": [int(c.strip()) for c in map_data[filename].split(",")],
            "label": label,
            "annotated_path": f"outputs/images/{filename}"
        }
        
        status = "🔴 危險（有火）" if is_danger else "🟢 安全（無火）"
        temp_str = f"{vision_temp:.1f}°C" if vision_temp is not None else "無火"
        print(f"   → {status} | 火焰溫度: {temp_str} | 決策: {suggested_action}")
    
    if not results:
        print("❌ 沒有成功處理任何圖片")
        exit(1)
    
    # 取得原始影像尺寸
    sample_data = next(iter(results.values()))
    sample_path = sample_data["annotated_path"]
    sample_img = cv2.imread(sample_path)
    if sample_img is None:
        sample_img = cv2.imread(os.path.join(package_dir, list(results.keys())[0]))
    
    orig_h, orig_w = sample_img.shape[:2]
    

    MAP_SIZE = 1000           
    MONITOR_WIDTH = 250      
    NUM_MONITORS_PER_ROW = 4  
    
    monitor_h = int(MONITOR_WIDTH * orig_h / orig_w)
    
    # 建立地圖（寬 1000 × 高 500）
    map_w = MAP_SIZE
    map_h = MAP_SIZE // 2
    map_img = np.ones((map_h, map_w, 3), dtype=np.uint8) * 255
    
    # 座標轉換（0~1000 絕對座標）
    scale_x = map_w / 1000.0
    scale_y = map_h / 1000.0
    
    # 地下室拓樸圖
    adj = {
        "left1": ["mid"],
        "mid": ["left1", "right1", "right2"],
        "right1": ["mid",],
        "right2": ["mid", ]
    }
    
    # 先畫連線（灰色）
    connection_pairs = [("left1", "mid"), ("mid", "right1"), ("mid", "right2")]
    for a, b in connection_pairs:
        ap = bp = None
        for data in results.values():
            if data["label"] == a:
                ax, ay = data["coord"]
                ap = (int(ax * scale_x), int(ay * scale_y))
            if data["label"] == b:
                bx, by = data["coord"]
                bp = (int(bx * scale_x), int(by * scale_y))
        if ap and bp:
            cv2.line(map_img, ap, bp, (180, 180, 180), 6)
    
    # 畫每個 camera 位置的點（綠=安全，紅=危險）
    for data in results.values():
        x, y = data["coord"]
        px = int(x * scale_x)
        py = int(y * scale_y)
        color = (0, 0, 255) if data["is_danger"] else (0, 255, 0)
        cv2.circle(map_img, (px, py), 22, color, -1)
        cv2.circle(map_img, (px, py), 27, (0, 0, 0), 4)
        cv2.putText(map_img, data["label"], (px + 35, py + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    
    # 計算逃生指示
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
    
    # === 最終合成圖（地圖 + 一排四張監視器 + 文字）===
    num_monitor_rows = (len(file_list) + NUM_MONITORS_PER_ROW - 1) // NUM_MONITORS_PER_ROW  # 未來支援 >4 張自動換行
    total_monitor_height = num_monitor_rows * monitor_h
    
    canvas_w = MAP_SIZE
    canvas_h = map_h + total_monitor_height + 520
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    # 1. 貼上地圖
    canvas[0:map_h, 0:map_w] = map_img
    
    # 2. 監視器畫面 
    order = ["left1.png", "mid.jpg", "right1.png", "right2.jpg"]
    monitor_offset_x = 0   # 因為 4*250 = 1000，正好填滿整行
    grid_y = map_h
    
    for idx, fn in enumerate(order):
        row = idx // NUM_MONITORS_PER_ROW
        col = idx % NUM_MONITORS_PER_ROW
        
        if fn in results:
            img = cv2.imread(results[fn]["annotated_path"])
            if img is None:
                img = np.ones((orig_h, orig_w, 3), dtype=np.uint8) * 100
            img = cv2.resize(img, (MONITOR_WIDTH, monitor_h))
            
            label = results[fn]["label"]
            status_str = "[Danger]" if results[fn]["is_danger"] else "[Safety]"
            cv2.putText(img, f"{label} {status_str}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 4)
            cv2.putText(img, f"{label} {status_str}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
        else:
            img = np.ones((monitor_h, MONITOR_WIDTH, 3), dtype=np.uint8) * 180
        
        x = monitor_offset_x + col * MONITOR_WIDTH
        y = grid_y + row * monitor_h
        canvas[y:y + monitor_h, x:x + MONITOR_WIDTH] = img
    
    # 3. 逃生指示文字
    text_y = map_h + total_monitor_height + 40
    cv2.putText(canvas, "Escape Instructions at each location", (80, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.35, (0, 0, 0), 3)
    
    y_offset = text_y + 70
    for label in ["left1", "mid", "right1", "right2"]:
        if label in escape_texts:
            txt = escape_texts[label]
            cv2.putText(canvas, f"{label} : {txt}", (80, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 2, cv2.LINE_AA)
            y_offset += 68
    
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/b1_composite_map.png"
    cv2.imwrite(output_path, canvas)
    
    print(f"\n✅ 模擬完成！合成地圖已儲存 → {output_path}")
    print(f"   地圖尺寸：{MAP_SIZE}x{map_h} | 監視器排版：一排 {NUM_MONITORS_PER_ROW} 張")
    print("   綠點 = 安全　｜　紅點 = 危險")
    print("   （可直接打開 outputs/b1_composite_map.png 查看完整結果）")