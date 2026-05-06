# HAI Interaction System - 동작 및 구조 분석 명세서

본 문서는 [main.py](file:///Users/kyl/Desktop/HAI_interaction/main.py) 실행 시 구동되는 손 인식 기반 가상 마우스 제어 시스템의 전체적인 코드 구조와 각 컴포넌트의 상세한 동작 원리를 설명합니다.

## 1. 시스템 개요 (System Overview)
이 시스템은 카메라(Intel RealSense 또는 일반 웹캠)를 통해 들어오는 영상에서 사용자의 손 관절(Landmark)을 실시간으로 추적하여, 컴퓨터의 마우스 커서를 제어하는 가상 마우스(Virtual Mouse) 애플리케이션입니다. 

**주요 기능:**
- **카메라 자동 전환 (Fallback):** RealSense 카메라가 연결되어 있으면 적외선(IR) 스트림을 사용하고, 기기가 없을 경우 일반 웹캠으로 자동 전환됩니다.
- **핸드 트래킹 (Hand Tracking):** MediaPipe를 사용하여 손바닥 중심 좌표(무게중심)를 안정적으로 계산합니다.
- **모드 전환 (Mode Switching):** 화면 우측 하단의 특정 구역(Zone)에 손을 3초간 올려두면 '마우스 제어 모드(VIRTUAL TOUCH)'와 '단순 추적 모드(HAND TRACKING)' 간에 전환됩니다.
- **마우스 부드러운 제어 (Smoothing):** PyAutoGUI를 통해 커서를 이동시키며, 이전 위치와의 보간(Smoothing)을 통해 손떨림을 방지합니다.

---

## 2. 모듈별 상세 분석 (Module Breakdown)

최근 객체 지향(OOP) 구조로 리팩토링되어, 각 기능이 독립적인 클래스 모듈로 명확하게 분리되어 있습니다. 프로그램 구동 시 아래와 같은 역할을 수행합니다.

### 2.1. [main.py](file:///Users/kyl/Desktop/HAI_interaction/main.py) (프로그램 진입점)
극도로 단순화된 애플리케이션의 진입점(Entry Point)입니다.
- [app.py](file:///Users/kyl/Desktop/HAI_interaction/app.py)에서 [InteractionApp](file:///Users/kyl/Desktop/HAI_interaction/app.py#9-130) 클래스를 임포트하고, 인스턴스화 한 뒤 `app.start()`를 호출하여 메인 생명주기를 시작합니다.

### 2.2. [app.py](file:///Users/kyl/Desktop/HAI_interaction/app.py) ([InteractionApp](file:///Users/kyl/Desktop/HAI_interaction/app.py#9-130) 클래스)
애플리케이션의 상태 관리 및 메인 루프(Main Loop)를 담당하는 심장부입니다.
- **초기화 ([__init__](file:///Users/kyl/Desktop/HAI_interaction/detector/landmark.py#5-17)):** 카메라([get_camera](file:///Users/kyl/Desktop/HAI_interaction/core/camera.py#83-94)), 마우스 컨트롤러([VirtualMouse](file:///Users/kyl/Desktop/HAI_interaction/core/mouse.py#7-28)), 핸드 추적기([Detector](file:///Users/kyl/Desktop/HAI_interaction/detector/landmark.py#4-52)), UI 매니저([UIManager](file:///Users/kyl/Desktop/HAI_interaction/core/ui.py#3-34)) 객체를 생성하고 시스템 전역 설정값을 초기화합니다.
- **실행 루프 ([run_loop](file:///Users/kyl/Desktop/HAI_interaction/app.py#46-130)):** 무한 루프 블록 안에서 1초에 수십 번씩 다음 작업을 반복합니다.
  1. **프레임 획득:** `self.camera.get_frame()`으로 카메라 이미지를 읽어옵니다.
  2. **추론 및 좌표 추출:** `self.detector.process_hands()`를 호출하여 손 관절을 분석하고, [get_left_hand_pos()](file:///Users/kyl/Desktop/HAI_interaction/detector/landmark.py#26-38)로 조작의 기준이 되는 손바닥 중심의 (x, y) 정규화 좌표를 얻어옵니다.
  3. **UI 렌더링:** `self.ui`의 정적 메서드들을 통해 화면에 손 관절 레이어 덧그리기, 모드 전환 구역(Zone) 박스 표시, 활성 모드 텍스트 등을 시각화합니다.
  4. **모드 전환 로직:** 손 좌표가 화면 우측 하단의 지정된 구역 영역(`zone_x`, `zone_y`) 내에 들어왔는지 지속적으로 확인합니다. 이 구간 안에서 코드가 타이머를 동작시키며, 3초(`time_to_mode_switch`)가 도달하면 `current_mode` 변수를 스위칭합니다.
  5. **마우스 제어 및 윈도우 팝업:** 현재 모드가 `virtual touch`라면, 도출된 손 좌표를 `self.mouse.move()`에 넘겨 마우스 커서를 제어합니다. 마지막으로 `cv2.imshow`로 렌더링된 결과를 사용자 모니터에 출력합니다.

### 2.3. [core/camera.py](file:///Users/kyl/Desktop/HAI_interaction/core/camera.py) (카메라 하드웨어 제어)
카메라 하드웨어와의 통신을 추상화(Abstraction) 하여, 애플리케이션이 실제 하드웨어 종류에 의존하지 않도록 돕습니다.
- **[RealSenseCamera](file:///Users/kyl/Desktop/HAI_interaction/core/camera.py#6-51) 클래스:**
  - `pyrealsense2` 라이브러리를 활용하여 3D 깊이 카메라와 통신합니다. 외부 조명의 방해에 강인하기 위해 기본 RGB 이미지가 아닌 단일 채널 적외선(Infrared, `y8` 형식) 스트림을 사용하도록 셋업합니다. 
  - 어두운 곳에서의 성능 향상을 위해 카메라 자체의 IR 발광기(IR Emitter)를 최대치로 강제 점등합니다.
- **[WebcamCamera](file:///Users/kyl/Desktop/HAI_interaction/core/camera.py#52-82) 클래스 (Fallback 기능):**
  - OpenCV(`cv2.VideoCapture`)를 사용하는 일반 웹캠 클래스입니다.
  - RealSense 구조와의 완벽한 호환을 맞추기 위해, 웹캠의 BGR 컬러 프레임을 강제로 흑백(`cv2.COLOR_BGR2GRAY`)으로 변환시켜 [app.py](file:///Users/kyl/Desktop/HAI_interaction/app.py)에 넘겨줍니다. 
- **[get_camera()](file:///Users/kyl/Desktop/HAI_interaction/core/camera.py#83-94) 팩토리 함수:**
  - 실행 시점에 주변 기기를 탐색하여 RealSense 기기가 존재하면 [RealSenseCamera](file:///Users/kyl/Desktop/HAI_interaction/core/camera.py#6-51)를, 없으면 [WebcamCamera](file:///Users/kyl/Desktop/HAI_interaction/core/camera.py#52-82) 객체를 다형성 원리에 입각하여 반환합니다. 

### 2.4. [core/mouse.py](file:///Users/kyl/Desktop/HAI_interaction/core/mouse.py) ([VirtualMouse](file:///Users/kyl/Desktop/HAI_interaction/core/mouse.py#7-28) 클래스)
운영체제(OS)의 실제 마우스 커서를 제어하고 좌표계를 매핑하는 로직입니다.
- **마우스 이벤트 제어:** 파이썬 환경의 `pyautogui` 모듈을 사용하여 화면 커서 위치를 강압적으로 통제합니다. 
- **ROI (Region of Interest) 동적 스케일링:** 사용자가 손을 화면의 끄트머리까지 뻗지 않아도 되도록, 카메라 전체 화면을 사용하지 않고 화면 가운데 일부 (0.3 ~ 0.7 범위 비율) 영역만 민감하게 반응할 수 있는 구역으로 정의하고, 이를 모니터 모서리와 1:1 매핑 시킵니다.
- **스무딩 필터 (Low-Pass Filter 로직):** 손 떨림에 민감하게 반응하지 않도록 이동 거리에 감쇠 계수(`smoothing = 0.5`)를 곱하여, 이전 좌표 좌표계와 현재 타겟 좌표계 사이를 부드럽게 보간(Interpolate)하여 미끄러지듯 이동합니다.

### 2.5. [core/ui.py](file:///Users/kyl/Desktop/HAI_interaction/core/ui.py) ([UIManager](file:///Users/kyl/Desktop/HAI_interaction/core/ui.py#3-34) 클래스)
OpenCV 프레임 이미지 위에 컴퓨터 그래픽 요소들을 덧그려주는 역할을 합니다.
- 색상 선언, 폰트 종류, 박스 그리기 좌표 등 UI/UX 관련 로직들을 별도의 공간으로 빼내어 응집도를 높였습니다. 정적 팩토리 패턴(`@staticmethod`)으로 구성되어 객체 생성 없이 메서드를 직접 사용합니다.

### 2.6. [detector/landmark.py](file:///Users/kyl/Desktop/HAI_interaction/detector/landmark.py) ([Detector](file:///Users/kyl/Desktop/HAI_interaction/detector/landmark.py#4-52) 클래스)
구글의 MediaPipe AI 모델을 감싸고 있는 추론 모듈입니다.
- **손 관절 추적:** `mp.solutions.hands` 모듈을 생성하여 RGB로 치환된 프레임 이미지를 분석하고 21개의 손가락 관절 좌표를 추출합니다.
- **무게중심 좌표 확보 (매우 중요):** 마우스 커서를 제어하기 위해 검지 손가락 끝점(Fingertip)과 같은 한 지점만 사용하면 다른 손마디를 움직일 때 기준선이 변하여 커서가 튀는 문제가 발생합니다. 이 문제를 해결하기 위해 **손목(0번), 검지~새끼 기저부(5, 9, 13, 17번) 등 핵심 관절 좌표점 5개의 평균 좌표(Centroid)**를 도출해 마우스 컨트롤 조종점으로 사용합니다.
- **손 판별 매핑 로직:** 웹캠이나 일반 카메라가 거울을 보듯 좌우가 바뀐 채로 입력되는 특성을 역산하여, 화면에 보이는 손이 사용자의 실제 어떤 손인지 올바르게 매치하여 반환합니다.

---

## 4. 프로그램 동작 주기 (Data Flow)

프로그램이 시작된 후 1초당 프레임워크 숫자(약 30번)만큼 다음 절차가 연속해서 반복 발생합니다.

1. **데이터 생산:** Camera 단말(RealSense or Webcam)이 현실 영역을 캡처하여 흑백의 NumPy 배열을 생성합니다.
2. **AI 검출:** [Detector](file:///Users/kyl/Desktop/HAI_interaction/detector/landmark.py#4-52)가 흑백 배열에서 손의 형상을 식별하고, 손바닥 중심의 x, y 상대 비율 좌표 (`0.0 ~ 1.0` 사이)를 알아냅니다.
3. **디스플레이 및 트리거 평가:** [UIManager](file:///Users/kyl/Desktop/HAI_interaction/core/ui.py#3-34)에 명령을 내려 화면에 그림을 덧칠하고, [app.py](file:///Users/kyl/Desktop/HAI_interaction/app.py)는 현재 손 좌표가 우측 하단 시스템 조작 영역 내부에 들어와 있는지 평가하고 시간을 측정합니다.
4. **마우스 투영:** 현재 시스템 펜딩 결과가 'VIRTUAL TOUCH' 활성 상태라면, 손바닥 중심(x, y) 비율을 [VirtualMouse](file:///Users/kyl/Desktop/HAI_interaction/core/mouse.py#7-28)로 전달. [VirtualMouse](file:///Users/kyl/Desktop/HAI_interaction/core/mouse.py#7-28)는 이를 모니터 디스플레이 픽셀 단위로 환산하고 이전 좌표 기준의 스무딩을 더하여 윈도우 OS단에 실제 `Mouse Move` 인터럽트를 보냅니다.
5. **화면 송출:** 최종적으로 덧칠이 끝난 이미지 배열이 구별 가능한 OpenCV 윈도우 그래픽 유저 인터페이스 창으로 브로드캐스트 됩니다. 이 과정은 사용자에게 즉각적이고 부드러운 반응을 제공합니다.
