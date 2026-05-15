#!/usr/bin/env python3
"""
3D 가상터치 그림판 - WebSocket 서버
React UI와 MediaPipe 손 인식을 연결합니다.
"""

import asyncio
import websockets
import json
import time

# 현재 손 데이터 (전역 변수)
current_hand_data = {
    'handedness': 'none',
    'status': 'idle',
    'position': {'x': 0, 'y': 0, 'z': 0},
    'gesture': 'open',
    'drawingPoints': []
}

# 연결된 클라이언트 목록
connected_clients = set()


async def handle_client(websocket, path):
    """클라이언트 연결 핸들러"""
    # 클라이언트 추가
    connected_clients.add(websocket)
    client_address = websocket.remote_address
    print(f"✅ 클라이언트 연결됨: {client_address}")
    print(f"📊 현재 연결된 클라이언트 수: {len(connected_clients)}")

    try:
        while True:
            # 현재 손 데이터를 JSON으로 전송
            message = json.dumps(current_hand_data)
            await websocket.send(message)

            # 60fps (16ms 간격)
            await asyncio.sleep(0.016)

    except websockets.exceptions.ConnectionClosed:
        print(f"❌ 클라이언트 연결 끊김: {client_address}")
    finally:
        # 클라이언트 제거
        connected_clients.remove(websocket)
        print(f"📊 현재 연결된 클라이언트 수: {len(connected_clients)}")


async def simulate_hand_movement():
    """
    테스트용: 손 움직임 시뮬레이션
    실제로는 MediaPipe에서 데이터를 받아 current_hand_data를 업데이트합니다.
    """
    import math

    t = 0
    while True:
        # 원형 움직임 시뮬레이션
        x = 320 + 200 * math.cos(t)
        y = 240 + 200 * math.sin(t)
        z = 50 + 30 * math.sin(t * 2)

        # 데이터 업데이트
        current_hand_data['handedness'] = 'right'
        current_hand_data['status'] = 'drawing' if t % 6 < 3 else 'idle'
        current_hand_data['position'] = {
            'x': int(x),
            'y': int(y),
            'z': int(z)
        }

        t += 0.1
        await asyncio.sleep(0.1)


async def main():
    """메인 서버 실행"""
    print("=" * 60)
    print("🚀 3D 가상터치 그림판 WebSocket 서버 시작")
    print("=" * 60)
    print(f"📡 서버 주소: ws://localhost:8765")
    print(f"🌐 React UI 연결 대기 중...")
    print(f"💡 테스트 모드: 손 움직임 시뮬레이션 활성화")
    print(f"⚠️  MediaPipe 연동 시 simulate_hand_movement() 비활성화 필요")
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

    # 테스트용 손 움직임 시뮬레이션 시작
    # MediaPipe 사용 시 이 라인을 주석 처리하세요
    asyncio.create_task(simulate_hand_movement())

    print("✅ 서버 준비 완료! React UI를 열어주세요.")
    print("🛑 종료하려면 Ctrl+C를 누르세요.")
    print()

    # 서버 계속 실행
    await server.wait_closed()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n")
        print("=" * 60)
        print("🛑 서버 종료됨")
        print("=" * 60)
