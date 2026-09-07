"""번들 구성 점검. 카메라를 열거나 창을 띄우지 않고 자산과 모델 로딩만 확인합니다.

창 없는 exe는 실패해도 화면에 아무것도 남지 않으므로, 결과를 파일로도 남깁니다.
실제 제스처 인식 성능이 아니라 "이 PC에서 프로그램이 뜰 수 있는가"만 봅니다.
"""

import os
from pathlib import Path
import platform
import sys
import traceback

from .resources import is_frozen, resource_path, resource_root


def report_path():
    directory = os.environ.get("LOCALAPPDATA")
    if not directory:
        return Path(sys.executable).resolve().parent / "selftest.log"
    return Path(directory) / "JestureService" / "selftest.log"


def check_assets():
    from .hand_tracker import DEFAULT_MODEL
    from .settings import DEFAULT_CHECKPOINT
    from .yolo_gate import DEFAULT_WEIGHTS

    missing = []
    for name, path in (("MediaPipe 손 모델", DEFAULT_MODEL), ("LSTM 체크포인트", DEFAULT_CHECKPOINT),
                       ("손모양 게이트 가중치", DEFAULT_WEIGHTS)):
        state = "OK" if path.is_file() else "없음"
        if not path.is_file():
            missing.append(name)
        yield f"  [{state}] {name}: {path}"
    if missing:
        raise FileNotFoundError("번들 자산 누락: " + ", ".join(missing))


def check_imports():
    import numpy
    import cv2
    import torch
    yield f"  [OK] numpy {numpy.__version__} / opencv {cv2.__version__}"
    yield f"  [OK] torch {torch.__version__} (CUDA {torch.cuda.is_available()})"
    from PySide6 import __version__ as pyside
    yield f"  [OK] PySide6 {pyside}"
    import ultralytics
    yield f"  [OK] ultralytics {ultralytics.__version__}"


def check_models(device):
    from .predictor import GesturePredictor
    from .settings import DEFAULT_CHECKPOINT
    from .yolo_gate import DEFAULT_WEIGHTS, GateConfig, YoloGate

    import numpy as np

    predictor = GesturePredictor(DEFAULT_CHECKPOINT, device)
    yield f"  [OK] LSTM 복원: 라벨 {predictor.labels}, seq_len {predictor.seq_len}"
    # 빈 프레임 한 장으로 실제 추론 경로까지 지나갑니다. 결과 해석용 의존성이
    # 번들에서 빠졌을 때 로딩만으로는 드러나지 않기 때문입니다.
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    frames = np.zeros((predictor.seq_len, 21, 3), dtype=np.float32)
    frames[:, :, :2] = .5
    frames[:, 9, 1] = .2  # 손목-중지 거리를 0이 아니게 만들어 크기 기준을 세웁니다.
    times = np.linspace(0., 1.2, predictor.seq_len, dtype=np.float64)
    label_id, probabilities = predictor.predict(frames, times, np.ones(predictor.seq_len, dtype=bool))
    yield f"  [OK] LSTM 추론 1회: {predictor.labels[label_id]} {probabilities[label_id]:.2f} (합성 입력)"
    gate = YoloGate(DEFAULT_WEIGHTS, device, GateConfig(imgsz=320))
    yield f"  [OK] 손모양 게이트 복원: {sorted(gate.names.values())} ({gate.device})"
    yield f"  [OK] 손모양 추론 1회: {gate.predict(blank)[0] or '검출 없음'} (빈 프레임)"
    # MediaPipe detector는 카메라 없이 만들 수 있습니다. 번들에서 가장 잘 깨지는 부분입니다.
    from .hand_tracker import HandTracker
    tracker = HandTracker()
    try:
        yield "  [OK] MediaPipe 손 검출기 생성"
    finally:
        tracker.close()


def run(device="cpu"):
    """점검 결과 줄들과 성공 여부를 돌려줍니다."""
    lines = [
        "Jesture 자체 점검",
        f"  실행 형태: {'번들(exe)' if is_frozen() else '저장소'} / 자산 기준 폴더: {resource_root()}",
        f"  Python {platform.python_version()} / {platform.platform()}",
        "",
        "번들 자산",
    ]
    passed = True
    for title, check in (("", check_assets), ("의존성", check_imports), ("모델 로딩", check_models)):
        if title:
            lines.extend(["", title])
        try:
            lines.extend(check(device) if check is check_models else check())
        except Exception:
            passed = False
            lines.extend(["  [실패]", *("  " + line for line in traceback.format_exc().splitlines())])
            break
    lines.extend(["", "결과: " + ("정상 — 프로그램을 실행할 수 있습니다." if passed
                                else "실패 — 위 오류를 확인하세요.")])
    return lines, passed


def emit(text):
    """창 없는 exe에서는 stdout이 없거나 cp949라 유니코드를 못 씁니다. 점검을 죽이지 않습니다."""
    stream = sys.stdout
    if stream is None:
        return
    try:
        stream.write(text + "\n")
        stream.flush()
    except (UnicodeEncodeError, OSError, ValueError):
        try:
            encoding = getattr(stream, "encoding", None) or "ascii"
            stream.buffer.write(text.encode(encoding, "replace") + b"\n")
            stream.buffer.flush()
        except Exception:
            pass  # 보고서 파일이 이미 남았으므로 출력 실패로 결과를 바꾸지 않습니다.


def main(device="cpu"):
    lines, passed = run(device)
    text = "\n".join(lines)
    # 콘솔 출력보다 파일이 우선입니다. 창 없는 실행에서는 이 파일이 유일한 결과입니다.
    try:
        path = report_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        text += f"\n\n보고서: {path}"
    except OSError as exc:
        text += f"\n\n보고서를 저장하지 못했습니다: {exc}"
    emit(text)
    return 0 if passed else 1
