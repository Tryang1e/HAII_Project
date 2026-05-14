# RealSense Python 개발 환경 설정 가이드 (MacBook M1/M2/M3)

이 문서는 MacBook M1(Apple Silicon) 환경에서 Intel RealSense D435 카메라를 사용하기 위한 Python 가상 환경(` .venv`) 및 시스템 설정 방법을 안내합니다.

## 1. 전제 조건
- **Homebrew**: 패키지 관리를 위해 필요합니다.
- **Python 3.11**: MediaPipe 0.10.x의 타입 힌팅 및 라이브러리 호환성을 위해 **Python 3.11** 사용이 필수적입니다.
- **USB 3.0 케이블/포트**: 고해상도(640x480 이상) 스트리밍을 위해 필수입니다.

## 2. 시스템 드라이버 설치
macOS에서 RealSense 하드웨어에 접근하기 위해 `librealsense` 라이브러리를 설치해야 합니다.
```bash
brew install librealsense
```

## 3. Python 가상 환경 설정
프로젝트 루트 디렉토리에서 다음 명령어를 실행하여 가상 환경을 구축하고 필요한 라이브러리를 설치합니다.

```bash
# 가상 환경 생성 (반드시 python3.11 지정)
python3.11 -m venv .venv

# 가상 환경 활성화
source .venv/bin/activate

# 필수 패키지 설치
pip install --upgrade pip setuptools wheel
pip install numpy opencv-python pyrealsense2-macosx mediapipe==0.10.9
```

> [!IMPORTANT]
> - **Apple Silicon (M1/M2/M3)** 맥북에서는 `mediapipe==0.10.9` 버전이 가장 안정적입니다.
> - 최신 버전 MediaPipe는 `solutions` 모듈 위치가 변경되었으므로, 코드 작성 시 `import mediapipe.python.solutions.pose`와 같이 **`python`** 경로를 포함해야 합니다.

## 4. 실행 방법
macOS의 보안 정책 및 USB 장치 제어를 위해 **`sudo`** 권한이 필요합니다. 가상 환경의 파이썬 실행 파일 경로를 직접 지정하여 실행합니다.

```bash
sudo .venv/bin/python3 main.py
```

## 5. 트러블슈팅
1. **Segmentation Fault (139)**: 실행 중 갑자기 종료된다면 카메라를 다시 연결하거나, `cv2.imshow` 관련 창이 제대로 뜨는지 확인하세요. (M1 Pro 이상 기종에서 간혹 발생할 수 있습니다.)
2. **'failed to set power state'**: USB 3.0 포트/케이블인지 확인하십시오. (시스템 리포트에서 5 Gb/s 확인 필수)
3. **ModuleNotFoundError**: 가상 환경이 활성화된 상태에서 `pip list`를 통해 `mediapipe`와 `pyrealsense2-macosx`가 있는지 확인하세요.
