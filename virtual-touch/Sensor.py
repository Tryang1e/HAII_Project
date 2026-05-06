import math

class GestureSensor:
    def __init__(self):
        # 손가락 끝과 DIP 관절(끝에서 두 번째) 인덱스 매핑 (MediaPipe 기준)
        # 0: Wrist
        # 4: Thumb tip, 3: Thumb IP, 2: Thumb MCP, 1: Thumb CMC
        # 8: Index tip, 7: Index DIP, 6: Index PIP, 5: Index MCP
        # 12: Middle tip, 11: Middle DIP, 10: Middle PIP, 9: Middle MCP
        # 16: Ring tip, 15: Ring DIP, 14: Ring PIP, 13: Ring MCP
        # 20: Pinky tip, 19: Pinky DIP, 18: Pinky PIP, 17: Pinky MCP
        self.finger_tips = [8, 12, 16, 20]
        self.finger_pips = [6, 10, 14, 18]

    def _get_distance(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def detect_gesture(self, hand_landmarks):
        """
        주어진 손의 랜드마크를 분석하여 특정 제스처를 반환합니다.
        손가락이 접혀있는지, 펴져있는지를 확인하여 제스처를 결정합니다.
        """
        if not hand_landmarks:
            return None

        # 1. 펴진 손가락 확인 (엄지 제외)
        fingers_open = []
        for tip, pip in zip(self.finger_tips, self.finger_pips):
            # 손가락 끝(tip)이 손가락 두 번째 관절(pip)보다 위에 있는지 (화면상 더 작은 y값)
            # 또는 손목(0) 기준 거리를 통해 판단할 수도 있습니다.
            # 가장 간단한 방법은 손가락 끝이 손목(0)에서 더 멀리 떨어져 있는지, 
            # 그리고 손가락이 굽혀져 안으로 들어왔는지(y좌표 비교)를 보는 것입니다.
            # 여기서는 Y축 좌표 기준으로 직관적인 위/아래 판단 (화면 상단이 0)
            if hand_landmarks[tip].y < hand_landmarks[pip].y:
                fingers_open.append(True)
            else:
                fingers_open.append(False)
                
        # 2. 엄지 손가락 확인 (x좌표 기준으로 바깥으로 향하는지 판단이 필요하지만 단순화)
        # 엄지 끝(4)이 엄지 관절(2)보다 위에 있는지 (단순 세로 판단)
        thumb_open = hand_landmarks[4].y < hand_landmarks[2].y
        
        # --- 제스처 판별 로직 ---
        
        # 모든 손가락이 펴진 경우: Open Hand (보통은 엄지 포함 5개, 여기서는 최소 4개 이상 펴짐)
        if fingers_open.count(True) >= 4:
            return "Open Hand"
            
        # 모든 손가락이 접힌 경우: Fist
        if fingers_open.count(True) == 0 and not thumb_open:
            return "Fist"
            
        # 검지와 중지만 펴진 경우: V-Sign
        if fingers_open[0] and fingers_open[1] and not fingers_open[2] and not fingers_open[3]:
            return "V-Sign"
            
        # 검지만 펴진 경우: Pointing
        if fingers_open[0] and fingers_open.count(True) == 1:
            return "Pointing"

        return "Unknown"
