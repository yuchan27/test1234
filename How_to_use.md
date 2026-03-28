# 引入格式
# Program : decision_engine.py

# 從檔案 safety_engine.py 中匯入 SafetyDecisionEngine 類別
from safety_engine import SafetyDecisionEngine

# 1. 初始化引擎 (設定 FPS 與 警報閾值)
engine = SafetyDecisionEngine(fps=30, alarm_threshold=0.75)

# 2. 準備符合格式的 JSON Payload (包含 YOLO txt 格式)
payload = {
    "context": {
        "timestamp": "2026-03-28T12:45:32.123Z",
        "frame_id": 1042
    },
    "perceptions": {
        "visual_objects": "0 0.767 0.288 0.036 0.054", # YOLO 格式字串
        "environmental_sensors": {
            "temperature_celsius": 25.4
        }
    }
}

# 3. 執行決策評估
result = engine.evaluate_payload(payload)

# 4. 取得結果
if result["status"] == "success":
    print(f"決策結果: {result['decision']['suggested_action']}")
    print(f"解析訊息: {result['explainability']['trace_message']}")