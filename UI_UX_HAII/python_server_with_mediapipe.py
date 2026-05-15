#!/usr/bin/env python3
"""
3D 가상터치 그림판 - MediaPipe 연동 WebSocket 서버
실제 손 인식 기능이 포함된 버전입니다.
"""

import asyncio
import websockets
import json
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 현재 손 데이터
current_hand_data = {
    'handedness': 'none',
    'status': 'idle',
    'position': {'x': 0, 'y': 0, 'z': 0},
    'gesture': 'open',
    'drawingPoints': []
}

# 연결된 클라이언트
connected_clients = set()

# 카메라 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def calculate_distance(point1, point2):
    """두 점 사이의 유클리디안 거리 계산"""
    return np.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2 +
        (point1.z - point2.z) ** 2
    )


def detect_gesture(hand_landmarks):
    """손 제스처 감지 (핀치, 주먹, 열린 손)"""
    # 엄지와 검지 끝 좌표
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]

    # 핀치 감지
    distance = calculate_distance(thumb_tip, index_tip)
    if distance < 0.05:
        return 'pinch', 'drawing'

    # 주먹 감지 (모든 손가락이 접혀있는지 확인)
    finger_tips = [8, 12, 16, 20]  # 검지, 중지, 약지, 새끼 끝
    finger_mcp = [5, 9, 13, 17]    # 각 손가락 뿌리

    folded_count = 0
    for tip, mcp in zip(finger_tips, finger_mcp):
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[mcp].y:
            folded_count += 1

    if folded_count >= 3:
        return 'fist', 'rotating'

    return 'open', 'idle'


def process_hand_data(results, frame_width, frame_height):
    """MediaPipe 결과를 처리하여 손 데이터 업데이트"""
    if results.multi_hand_landmarks and results.multi_handedness:
        # 첫 번째 손만 처리 (추후 양손 처리 가능)
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness_info = results.multi_handedness[0]

        # 손 좌우 판별
        handedness = handedness_info.classification[0].label.lower()
        current_hand_data['handedness'] = handedness

        # 검지 끝 좌표
        index_tip = hand_landmarks.landmark[8]
        current_hand_data['position'] = {
            'x': int(index_tip.x * frame_width),
            'y': int(index_tip.y * frame_height),
            'z': int(index_tip.z * 100)  # 스케일 조정
        }

        # 제스처 감지
        gesture, status = detect_gesture(hand_landmarks)
        current_hand_data['gesture'] = gesture
        current_hand_data['status'] = status

    else:
        # 손이 감지되지 않음
        current_hand_data['handedness'] = 'none'
        current_hand_data['status'] = 'idle'


async def handle_client(websocket, path):
    """클라이언트 연결 핸들러"""
    connected_clients.add(websocket)
    client_address = websocket.remote_address
    print(f"✅ 클라이언트 연결됨: {client_address}")
    print(f"📊 현재 연결된 클라이언트 수: {len(connected_clients)}")

    try:
        while True:
            message = json.dumps(current_hand_data)
            await websocket.send(message)
            await asyncio.sleep(0.016)  # 60fps

    except websockets.exceptions.ConnectionClosed:
        print(f"❌ 클라이언트 연결 끊김: {client_address}")
    finally:
        connected_clients.remove(websocket)
        print(f"📊 현재 연결된 클라이언트 수: {len(connected_clients)}")


async def capture_hands():
    """카메라에서 손 인식 처리"""
    print("📹 카메라 시작...")

    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️  카메라 프레임 읽기 실패")
            await asyncio.sleep(0.1)
            continue

        # 좌우 반전 (거울 모드)
        frame = cv2.flip(frame, 1)

        # BGR을 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe 손 인식
        results = hands.process(rgb_frame)

        # 손 데이터 처리
        process_hand_data(results, frame.shape[1], frame.shape[0])

        # 손 랜드마크 그리기
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # 상태 표시
        cv2.putText(
            frame,
            f"Status: {current_hand_data['status']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"Hand: {current_hand_data['handedness']}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # 화면 표시
        cv2.imshow('MediaPipe Hands', frame)

        # 'q' 키로 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        await asyncio.sleep(0.001)  # 비동기 처리


async def main():
    """메인 서버 실행"""
    print("=" * 60)
    print("🚀 3D 가상터치 그림판 WebSocket 서버 (MediaPipe)")
    print("=" * 60)
    print(f"📡 서버 주소: ws://localhost:8765")
    print(f"📹 카메라 활성화 중...")
    print(f"🌐 React UI 연결 대기 중...")
    print("=" * 60)
    print()

    # WebSocket 서버 시작
    server = await websockets.serve(
        handle_client,
        "localhost",
        8765,
        ping_interval=20,
        ping_timeout=10
    )

    print("✅ 서버 준비 완료!")
    print("🛑 종료하려면 카메라 창에서 'q'를 누르거나 Ctrl+C를 누르세요.")
    print()

    # 손 인식 시작
    await capture_hands()

    # 정리
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n")
        print("=" * 60)
        print("🛑 서버 종료됨")
        print("=" * 60)
        cap.release()
        cv2.destroyAllWindows()
