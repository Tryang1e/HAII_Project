# [기획서] 3D 가상 터치 그림판 (Human-AI Interaction)

## 1. 프로젝트 개요

* [cite_start]**프로젝트명**: 3D 가상 터치 그림판 [cite: 3, 7]
* [cite_start]**교과목명**: 휴먼AI인터랙션 (담당교수: 이건영) [cite: 3]
* [cite_start]**소속**: IT대학 컴퓨터공학과 2조 [cite: 3]
* **팀 구성**:
  * [cite_start]**김민수**: 프로덕트 총괄 (목표, 일정, 지표 관리 및 조율) [cite: 3, 10]
  * [cite_start]**최지민/이재민**: 개발 및 UI/UX 제작 (핸드 트래킹, Z축 추정, 제스처 인식, 가상 팔레트 설계) [cite: 10]
  * [cite_start]**한아름**: 시스템 통합 (파이프라인 연결 및 성능 최적화) [cite: 10]
* [cite_start]**핵심 컨셉**: 별도의 장비 없이 카메라만을 이용하여 허공에 손짓하는 것만으로 3D 공간에 그림을 그리거나 오브젝트를 조작하는 인터페이스 [cite: 14]

---

## 2. 주요 기능 및 기술 사양

### 2.1 핵심 기능 (Core Functions)

* [cite_start]**3D 드로잉**: 실시간 손가락 끝 위치 추적(Hand Landmark Tracking)과 가상 Z축(깊이) 인식을 결합한 3차원 선 표현 [cite: 16, 94]
* [cite_start]**가역성 지원**: 작업 실수나 오인식을 즉각 복구할 수 있는 Undo/Redo 기능 [cite: 19, 94]
* [cite_start]**캔버스 스크롤**: Grab & Drag(주먹을 쥐고 이동) 제스처를 통한 화면 이동 [cite: 94, 103]
* [cite_start]**UI 인터랙션**: Pinch(검지-엄지 맞댐) 제스처를 이용한 메뉴 선택 및 모드 전환 [cite: 94, 103]
* [cite_start]**모드 관리**: '캔버스 모드'(드로잉)와 '에디터 모드'(편집)를 분리하여 사용자 편의성 제공 [cite: 20]

### 2.2 사용 기술 스택 (Tech Stack)

* [cite_start]**언어**: Python (FSM 설계 및 시스템 개발) [cite: 91]
* [cite_start]**프레임워크**: MediaPipe Hands (실시간 랜드마크 추출) [cite: 91]
* [cite_start]**영상 처리/UI**: OpenCV (프레임 획득 및 출력), PyQt (GUI 구현) [cite: 91]
* [cite_start]**데이터 보정**: NumPy (Z축 연산), Pykalman (Kalman Filter 기반 노이즈 필터링) [cite: 91]

---

## 3. 인터랙션 및 시스템 설계

### 3.1 설계 원칙

* [cite_start]**인간공학적 최적화**: 피츠의 법칙(Fitts's Law)을 적용하여 메뉴 타겟 크기 최적화 및 이동 거리 단축 [cite: 97]
* [cite_start]**오입력 방지**: FSM(상태 제어)을 통해 드로잉과 UI 조작 상태를 분리하여 간섭 차단 [cite: 98, 111]
* [cite_start]**피로도 감소**: '고릴라 암' 현상을 고려하여 손의 이동 범위를 최소화하는 위치에 핵심 기능 배치 [cite: 18, 34]

### [cite_start]3.2 시스템 아키텍처 [cite: 83-87]

1. **입력층**: 웹캠을 통한 영상 프레임 획득 및 전처리
2. **분석층**: MediaPipe 기반 (x, y, z) 좌표 계산, Kalman Filter 적용, FSM 기반 상태 제어
3. **렌더링층**: 사용자 시점 보정 및 3D 스트로크/UI 실시간 시각화

---

## 4. 목표 성능 및 테스트 계획

* [cite_start]**응답 속도**: 제스처 인터랙션 후 시각적 피드백까지 지연 시간 1초 미만 (실시간 렌더링 50ms 미만) [cite: 25]
* [cite_start]**인식 정확도**: 제스처 트리거 판단 오류 5% 미만, 전체 인식 성공률 90% 이상 달성 [cite: 25]
* [cite_start]**학습 용이성**: 전체 핵심 기능을 파악하는 데 걸리는 시간 3분 이내 [cite: 25]

---

# [AI 학습용] 프롬프트 최적화 데이터 요약

AI가 이 프로젝트의 맥락을 이해하고 관련 코드나 설계를 제안할 때 참조하기 좋은 요약본입니다.

### 1. Context & Goal

* **Domain**: NUI(Natural User Interface), Computer Vision, 3D Canvas.
* **Objective**: 웹캠 기반 비접촉 3D 드로잉 시스템 구현.
* **Constraints**: 별도 하드웨어(VR 컨트롤러 등) 사용 금지, 일반 웹캠 환경에서의 Z축 인식 정확도 확보 필요.

### 2. Gesture Dictionary (Mapping)

| 제스처명              | 동작 설명             | 시스템 의미/기능                    |
| :-------------------- | :-------------------- | :---------------------------------- |
| **Hovering**    | 검지를 핀 상태로 이동 | 포인터 추적 및 위치 탐색            |
| **Pinch Click** | 엄지와 검지 끝을 맞댐 | 드로잉 일시 중지/재개, UI 버튼 클릭 |
| **Grab & Drag** | 주먹을 쥔 채로 이동   | 캔버스 스크롤 및 이동               |

### 3. Technical Logic for AI Implementation

* **Noise Reduction**: Kalman Filter를 사용하여 웹캠 하드웨어 노이즈 및 손떨림 억제.
* **Depth Estimation**: 2D 랜드마크 기반 가상 Z축 산출 알고리즘 적용.
* **State Management**: FSM을 사용하여 Idle, Drawing, Menu Selection 상태를 명확히 구분하여 설계.
* **UI/UX**: 피츠의 법칙을 활용한 버튼 배치 및 시각적 피드백(Highlight, Progress bar) 제공.

### 4. Problem Solving Strategy

* **Gorilla Arm**: 제스처 이동 범위 최소화 설계.
* **Inadvertent Input**: UI 영역 점유 시 Pinch 이벤트를 UI에 우선 할당하거나, 특정 모드에서만 드로잉 활성화.

git test
