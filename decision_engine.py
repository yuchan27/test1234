import json
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from collections import deque
from datetime import datetime

class SafetyDecisionEngine:
    """
    神經符號安全決策引擎模組
    負責接收標準化 JSON Payload，進行多模態特徵解析、模糊規則匹配與衝突解決。
    """
    def __init__(self, fps=30, alarm_threshold=0.75):
        self.fps = fps
        self.alarm_threshold = alarm_threshold
        # 滑動時間窗緩衝區 (保存歷史溫度與精確的 Unix Timestamp)
        self.temp_buffer = deque(maxlen=fps * 2) 
        
        # 載入 Fuzzy 規則庫
        self._initialize_fuzzy_rules()

    def _initialize_fuzzy_rules(self):
        """建立防幻覺與多模態融合的 Fuzzy 引擎 (大腦)"""
        self.v_conf = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'v_conf')
        self.t_grad = ctrl.Antecedent(np.arange(0, 5.1, 0.1), 't_grad')
        self.w_v = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'w_v')

        self.v_conf.automf(3, names=['low', 'medium', 'high'])
        self.t_grad['zero'] = fuzz.trimf(self.t_grad.universe, [0, 0, 0.6])
        self.t_grad['high'] = fuzz.trimf(self.t_grad.universe, [0.5, 5.0, 5.0])
        
        self.w_v['very_low'] = fuzz.trimf(self.w_v.universe, [0, 0, 0.3]) 
        self.w_v['medium'] = fuzz.trimf(self.w_v.universe, [0.2, 0.5, 0.8]) 
        self.w_v['high'] = fuzz.trimf(self.w_v.universe, [0.7, 1.0, 1.0]) 

        # 核心防呆規則
        rule1 = ctrl.Rule(self.v_conf['high'] & self.t_grad['zero'], self.w_v['very_low']) # 圖片攻擊
        rule2 = ctrl.Rule(self.v_conf['high'] & self.t_grad['high'], self.w_v['high'])     # 真實火災
        rule3 = ctrl.Rule(self.v_conf['low'] & self.t_grad['high'], self.w_v['high'])      # 溫度計故障
        rule4 = ctrl.Rule((self.v_conf['low'] | self.v_conf['medium']) & self.t_grad['zero'], self.w_v['medium']) # 常態

        self.w_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
        self.w_sim = ctrl.ControlSystemSimulation(self.w_ctrl)

    def evaluate_payload(self, json_payload):
        """
        公開介面：嚴謹解析輸入的 JSON，並輸出最終決策
        """
        try:
            # 1. 資料解析 (Data Ingestion) ------------------------
            data = json.loads(json_payload) if isinstance(json_payload, str) else json_payload
            
            # 1a. 解析 Context (處理 ISO 8601 時間戳記)
            time_str = data['context']['timestamp']
            # 將結尾的 'Z' 替換為 '+00:00' 讓 Python datetime 能安全轉換
            time_obj = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            timestamp_float = time_obj.timestamp()

            # 1b. 解析 感測器資料 (Sensors)
            sensors = data['perceptions']['environmental_sensors']
            current_temp = sensors['temperature_celsius']

            # 1c. 解析 視覺物件 (Visual Objects)
            visual_objects = data['perceptions']['visual_objects']
            v_conf = 0.0
            
            # 防呆遍歷：不預設物件順序，只尋找有危害標籤的物件
            for obj in visual_objects:
                if obj['label'] == 'fire':
                    # 如果有多個火災框，取信心度最高者
                    v_conf = max(v_conf, obj['confidence'])
                    
            # ----------------------------------------------------

        except KeyError as e:
            return self._generate_error_response(f"JSON 結構缺失必要欄位: {str(e)}")
        except Exception as e:
            return self._generate_error_response(f"JSON 解析發生例外錯誤: {str(e)}")

        # 2. 時序特徵工程 (計算溫度變化率 t_grad)
        self.temp_buffer.append((timestamp_float, current_temp))
        t_grad_val = 0.0
        
        if len(self.temp_buffer) > 5:
            t_start, temp_start = self.temp_buffer[0]
            t_end, temp_end = self.temp_buffer[-1]
            dt = t_end - t_start
            
            if dt > 0:
                t_grad_val = max(0.0, min(5.0, (temp_end - temp_start) / dt))

        # 3. 執行 Rule Matching 與權重推論
        if v_conf < 0.1:
            w_v_val = 0.5 
        else:
            self.w_sim.input['v_conf'] = v_conf
            self.w_sim.input['t_grad'] = t_grad_val
            try:
                self.w_sim.compute()
                w_v_val = self.w_sim.output['w_v']
            except:
                w_v_val = 0.1 
            
        w_s_val = 1.0 - w_v_val
        
        # 4. 決策層級融合 (Decision Fusion)
        temp_norm = max(0.0, min(1.0, (current_temp - 25.0) / 35.0))
        final_risk = (w_v_val * v_conf) + (w_s_val * temp_norm)
        
        # 5. 生成標準化輸出
        return self._generate_decision_output(final_risk, w_v_val, w_s_val, current_temp, v_conf)

    def _generate_decision_output(self, final_risk, w_v, w_s, temp, conf):
        """內部私有方法：生成送往 Action Execution 的決策字典"""
        is_alarm = final_risk >= self.alarm_threshold
        
        trace_msg = "狀態監控中..."
        if is_alarm:
            trace_msg = f"🔴 觸發火災警報！綜合風險 {final_risk:.2f}。影像權重 {w_v:.2f}，感測器權重 {w_s:.2f}。"
        elif conf > 0.8 and final_risk < self.alarm_threshold:
            trace_msg = f"🟢 攔截疑似圖片假警報。視覺特徵雖高 ({conf:.2f}) 但環境溫度未異常，視覺權重已強制降至 {w_v:.2f}。"
        elif temp > 60.0 and final_risk < self.alarm_threshold:
             trace_msg = f"🟡 攔截疑似硬體故障。感測器溫度異常 ({temp:.1f}°C) 但無起火視覺特徵，感測器權重已強制降至 {w_s:.2f}。"

        return {
            "status": "success",
            "decision": {
                "trigger_alarm": bool(is_alarm),
                "risk_score": float(final_risk),
                "suggested_action": "EVACUATE_AND_SHUTDOWN" if is_alarm else "CONTINUE_MONITORING"
            },
            "explainability": {
                "trace_message": trace_msg,
                "internal_weights": {"vision_weight": round(float(w_v), 3), "sensor_weight": round(float(w_s), 3)}
            }
        }

    def _generate_error_response(self, error_msg):
        return {
            "status": "error",
            "error_message": error_msg,
            "decision": {"trigger_alarm": False, "risk_score": 0.0, "suggested_action": "CHECK_SYSTEM"}
        }