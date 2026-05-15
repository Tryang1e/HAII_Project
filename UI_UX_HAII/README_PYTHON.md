# Python 서버 실행 가이드

## 빠른 시작

### 1. Python 패키지 설치

```bash
pip install -r requirements.txt
```

또는 개별 설치:

```bash
pip install websockets mediapipe opencv-python numpy
```

---

## 2. 서버 실행 방법

### 옵션 A: 테스트 서버 (MediaPipe 없이)

손 움직임을 시뮬레이션하는 간단한 테스트 서버입니다.

```bash
python python_server.py
```

**특징:**
- MediaPipe 불필요
- 자동으로 원형 움직임 시뮬레이션
- UI 테스트에 적합

---

### 옵션 B: 실제 손 인식 서버 (MediaPipe)

실제 카메라로 손을 인식하는 서버입니다.

```bash
python python_server_with_mediapipe.py
```

**특징:**
- 웹캠 필요
- MediaPipe로 실시간 손 인식
- 핀치, 주먹, 열린 손 제스처 감지
- 카메라 화면에서 'q' 키로 종료

---

## 3. 연결 확인

서버가 실행되면:

1. **콘솔 메시지 확인**
   ```
    3D 가상터치 그림판 WebSocket 서버 시작
    서버 주소: ws://localhost:8765
    서버 준비 완료! React UI를 열어주세요.
   ```

2. **React UI 확인**
   - 우측 하단에 "Python 연결됨" 메시지 표시
   - 초록색 점이 깜빡임

3. **손 인식 확인** (MediaPipe 서버 사용 시)
   - 카메라 창이 열림
   - 손을 비추면 랜드마크가 표시됨
   - UI에서 "오른손 인식됨" 등의 메시지 표시

---

## 4. 문제 해결

### 카메라가 열리지 않음 (MediaPipe)

```python
# python_server_with_mediapipe.py 파일 수정
cap = cv2.VideoCapture(1)  # 0 → 1로 변경 (다른 카메라 사용)
```

### 포트가 이미 사용 중

다른 프로그램이 8765 포트를 사용 중입니다.

**해결 방법:**
1. 다른 Python 서버 프로세스 종료
2. 또는 포트 변경:

```python
# 서버 파일에서 포트 변경
await websockets.serve(handle_client, "localhost", 9000)  # 8765 → 9000
```

React UI도 변경:
```typescript
// src/app/App.tsx
const { isConnected, handData } = useWebSocket('ws://localhost:9000');
```

### MediaPipe 설치 오류

Python 버전이 3.8-3.11인지 확인:

```bash
python --version
```

버전이 맞지 않으면 적절한 Python 설치 후 재시도

---

## 5. 데이터 구조

Python 서버가 보내는 JSON 형식:

```json
{
  "handedness": "right",
  "status": "drawing",
  "position": {
    "x": 320,
    "y": 240,
    "z": 50
  },
  "gesture": "pinch",
  "drawingPoints": []
}
```

---

## 6. 커스터마이징

### 전송 속도 조정

```python
# 60fps → 30fps로 변경
await asyncio.sleep(0.033)  # 0.016 → 0.033
```

### 손 개수 변경

```python
hands = mp_hands.Hands(
    max_num_hands=2,  # 1 → 2 (양손 인식)
    # ...
)
```

### 제스처 감도 조정

```python
# 핀치 감도 조정
if distance < 0.08:  # 0.05 → 0.08 (더 느슨하게)
    return 'pinch', 'drawing'
```

---

## 다음 단계

서버가 실행되면:

1. React UI에서 "Python 연결됨" 확인
2. 손을 움직여 상태 표시 확인
3. 캔버스에 실제 드로잉 로직 구현
4. 회전 행렬 적용
5. 미니맵 업데이트

