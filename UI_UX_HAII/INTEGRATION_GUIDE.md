# Python MediaPipe와 React UI 연결 가이드

## 개요
이 문서는 외부 Python 프로그램(MediaPipe 손 인식)과 React UI를 WebSocket으로 연결하는 방법을 설명합니다.

## 아키텍처

```
Python (MediaPipe)  ←→  WebSocket Server  ←→  React UI
     손 인식              localhost:8765         웹 브라우저
```

---

## 1. Python 쪽 설정

### 필요한 라이브러리 설치

```bash
pip install websockets
```

### WebSocket 서버 코드 추가

기존 MediaPipe 코드에 다음을 추가하세요:

```python
import asyncio
import websockets
import json
import threading

# 전역 변수로 손 데이터 저장
current_hand_data = {
    'handedness': 'none',
    'status': 'idle',
    'position': {'x': 0, 'y': 0, 'z': 0},
    'gesture': 'open',
    'drawingPoints': []
}

# WebSocket 서버 핸들러
async def handle_client(websocket, path):
    print(f"클라이언트 연결됨: {websocket.remote_address}")
    try:
        while True:
            # 현재 손 데이터를 JSON으로 전송
            message = json.dumps(current_hand_data)
            await websocket.send(message)
            await asyncio.sleep(0.016)  # 60fps (16ms)
    except websockets.exceptions.ConnectionClosed:
        print("클라이언트 연결 끊김")

# WebSocket 서버 시작
async def start_websocket_server():
    server = await websockets.serve(handle_client, "localhost", 8765)
    print("WebSocket 서버 시작: ws://localhost:8765")
    await server.wait_closed()

# 별도 스레드에서 WebSocket 서버 실행
def run_websocket_server():
    asyncio.run(start_websocket_server())

# MediaPipe 메인 루프에서 호출
if __name__ == "__main__":
    # WebSocket 서버를 별도 스레드로 시작
    ws_thread = threading.Thread(target=run_websocket_server, daemon=True)
    ws_thread.start()
    
    # MediaPipe 손 인식 루프
    while True:
        # 손 인식 결과를 current_hand_data에 업데이트
        # 예시:
        # current_hand_data['handedness'] = 'right'
        # current_hand_data['status'] = 'drawing'
        # current_hand_data['position'] = {'x': hand_x, 'y': hand_y, 'z': hand_z}
        pass
```

---

## 2. 데이터 형식 (JSON)

Python에서 React로 보낼 데이터 형식:

```json
{
  "handedness": "right",        // 'left', 'right', 'both', 'none'
  "status": "drawing",          // 'idle', 'drawing', 'rotating', 'zooming'
  "position": {
    "x": 320,
    "y": 240,
    "z": 50
  },
  "gesture": "pinch",           // 'pinch', 'fist', 'open'
  "drawingPoints": [            // 그려진 점들의 배열 (선택사항)
    {"x": 100, "y": 150, "z": 20},
    {"x": 105, "y": 155, "z": 22}
  ]
}
```

---

## 3. MediaPipe 데이터 매핑 예시

### 손 인식 → 상태 매핑

```python
import mediapipe as mp

# MediaPipe 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# 손 데이터 처리
def process_hand_landmarks(results):
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # 손 좌우 판별
            handedness = results.multi_handedness[idx].classification[0].label.lower()
            current_hand_data['handedness'] = handedness
            
            # 검지 끝 좌표
            index_tip = hand_landmarks.landmark[8]
            current_hand_data['position'] = {
                'x': int(index_tip.x * frame_width),
                'y': int(index_tip.y * frame_height),
                'z': int(index_tip.z * 100)  # 스케일 조정
            }
            
            # 제스처 판별 (핀치, 주먹 등)
            thumb_tip = hand_landmarks.landmark[4]
            distance = calculate_distance(thumb_tip, index_tip)
            
            if distance < 0.05:
                current_hand_data['gesture'] = 'pinch'
                current_hand_data['status'] = 'drawing'
            elif is_fist(hand_landmarks):
                current_hand_data['gesture'] = 'fist'
                current_hand_data['status'] = 'rotating'
            else:
                current_hand_data['gesture'] = 'open'
                current_hand_data['status'] = 'idle'
    else:
        current_hand_data['handedness'] = 'none'
        current_hand_data['status'] = 'idle'
```

---

## 4. React UI 실행

### 개발 서버 시작

React 프로젝트는 이미 실행 중입니다. WebSocket 연결 상태는 화면 우측 하단에 표시됩니다.

- **초록색**: Python 연결됨
- **빨간색**: Python 연결 끊김

---

## 5. 테스트 방법

### 1단계: Python WebSocket 서버 실행
```bash
python your_mediapipe_script.py
```

### 2단계: 연결 확인
- React UI 우측 하단에 "Python 연결됨" 메시지가 표시되면 성공
- 왼쪽 상단에 현재 상태(드로잉 중, 회전 중 등)가 표시됨

### 3단계: 손 인식 테스트
- 손을 카메라에 비추면 손 인식 상태가 실시간으로 표시됨
- 핀치 제스처 → "드로잉 중" 표시
- 주먹 제스처 → "회전 중" 표시

---

## 6. 문제 해결

### WebSocket 연결 안 됨
- Python 서버가 실행 중인지 확인
- 포트 8765가 사용 가능한지 확인
- 방화벽 설정 확인

### 데이터가 전송되지 않음
- `current_hand_data`가 올바르게 업데이트되는지 확인
- JSON 형식이 올바른지 확인
- 콘솔 로그 확인

### 프레임 드롭 발생
- WebSocket 전송 주기 조정 (0.016초 → 0.033초)
- 데이터 압축 고려

---

## 7. 다음 단계

- [ ] 캔버스에 실제 3D 라인 렌더링
- [ ] Undo/Redo 스택 구현
- [ ] 미니맵에 실시간 드로잉 반영
- [ ] 칼만 필터 적용 (손떨림 보정)
- [ ] 회전 행렬 동기화

---

## 참고 자료

- [WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [Python websockets 라이브러리](https://websockets.readthedocs.io/)
