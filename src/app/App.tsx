import { useState, useEffect } from 'react';
import { TopToolbar } from './components/TopToolbar';
import { LeftSidebar } from './components/LeftSidebar';
import { RightMinimap } from './components/RightMinimap';
import { CanvasArea } from './components/CanvasArea';
import { StatusIndicator } from './components/StatusIndicator';
import { ConnectionGuide } from './components/ConnectionGuide';
import { useWebSocket } from './hooks/useWebSocket';

export default function App() {
  // WebSocket 연결 (Python MediaPipe와 통신)
  const { isConnected, handData, connectionError, reconnect } = useWebSocket('ws://localhost:8765');

  // 도구 상태
  const [selectedTool, setSelectedTool] = useState<'brush' | 'eraser'>('brush');
  const [selectedColor, setSelectedColor] = useState('#FF0000');
  const [brushSize, setBrushSize] = useState(4);

  // 모드 상태
  const [mode, setMode] = useState<'2d' | '3d'>('3d');
  const [isRotateMode, setIsRotateMode] = useState(false);

  // 상태 표시 (WebSocket에서 받은 데이터로 업데이트)
  const [status, setStatus] = useState<'idle' | 'drawing' | 'rotating' | 'zooming'>('idle');
  const [handedness, setHandedness] = useState<'left' | 'right' | 'both' | 'none'>('none');

  // 연결 가이드 표시 여부
  const [showGuide, setShowGuide] = useState(false);

  // WebSocket 데이터 수신 시 상태 업데이트
  useEffect(() => {
    if (handData) {
      setStatus(handData.status);
      setHandedness(handData.handedness);
    }
  }, [handData]);

  // 편집 기능 핸들러
  const handleUndo = () => {
    console.log('Undo');
    // TODO: Undo 로직 구현
  };

  const handleRedo = () => {
    console.log('Redo');
    // TODO: Redo 로직 구현
  };

  const handleClear = () => {
    console.log('Clear');
    // TODO: Clear 로직 구현
  };

  const handleRotateModeToggle = () => {
    setIsRotateMode(!isRotateMode);
    if (!isRotateMode) {
      setStatus('rotating');
    } else {
      setStatus('idle');
    }
  };

  return (
    <div className="size-full flex flex-col bg-gray-950">
      {/* 상단 툴바 */}
      <TopToolbar
        selectedTool={selectedTool}
        onToolChange={setSelectedTool}
        selectedColor={selectedColor}
        onColorChange={setSelectedColor}
        brushSize={brushSize}
        onBrushSizeChange={setBrushSize}
        onUndo={handleUndo}
        onRedo={handleRedo}
        onClear={handleClear}
      />

      {/* 메인 영역 */}
      <div className="flex-1 flex overflow-hidden">
        {/* 왼쪽 사이드바 */}
        <LeftSidebar
          mode={mode}
          onModeChange={setMode}
          isRotateMode={isRotateMode}
          onRotateModeToggle={handleRotateModeToggle}
        />

        {/* 캔버스 영역 */}
        <CanvasArea mode={mode} />

        {/* 오른쪽 미니맵 */}
        <RightMinimap />
      </div>

      {/* 상태 표시 */}
      <StatusIndicator status={status} handedness={handedness} />

      {/* 연결 가이드 */}
      {showGuide && <ConnectionGuide isConnected={isConnected} onDismiss={() => setShowGuide(false)} />}

      {/* WebSocket 연결 상태 */}
      <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 items-end">
        <div
          className={`px-4 py-2 rounded-lg shadow-lg flex items-center gap-2 ${
            isConnected
              ? 'bg-green-600 text-white'
              : 'bg-gray-700 text-gray-300'
          }`}
        >
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-white animate-pulse' : 'bg-gray-400'}`} />
          <span className="text-sm font-medium">
            {isConnected ? 'Python 연결됨' : 'Python 대기 중'}
          </span>
          {!isConnected && (
            <>
              <button
                onClick={reconnect}
                className="ml-2 px-2 py-1 bg-gray-600 hover:bg-gray-500 rounded text-xs transition-colors"
              >
                재연결
              </button>
              <button
                onClick={() => setShowGuide(true)}
                className="ml-1 px-2 py-1 bg-blue-600 hover:bg-blue-500 rounded text-xs transition-colors"
              >
                ?
              </button>
            </>
          )}
        </div>

        {connectionError && !isConnected && (
          <div className="bg-yellow-600 text-white px-3 py-2 rounded-lg shadow-lg text-xs max-w-xs">
            💡 Python 서버를 먼저 실행하세요
          </div>
        )}
      </div>
    </div>
  );
}