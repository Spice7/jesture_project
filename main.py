"""Jesture 단일 진입점. 인자 없이 실행하면 통합 제스처 UI를 시작한다."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
COMMANDS = {
    "app": "scripts/gesture_app.py",
    "yolo": "yolo/yolo.py",
    "collect": "programs/collect_gesture.py",
    "collect-video": "programs/collect_gesture_video.py",
    "extract": "programs/extract_gesture_videos.py",
    "validate": "programs/validate_gesture_dataset.py",
    "review": "programs/data_preprocessing.py",
    "frames": "programs/slicing.py",
    "record": "util/slice.py",
    "flip": "programs/flip_image.py",
    "train-dynamic": "scripts/train_model.py",
    "eval-dynamic": "scripts/eval_model.py",
    "compare": "scripts/compare_models.py",
    "replay": "scripts/replay_segments.py",
    "inspect-npz": "util/valid_npz.py",
}
INTERACTIVE_COMMANDS = {
    "collect-video": "영상 경로와 라벨을 입력받아 동적 NPZ 수집",
    "extract": "videos/의 라벨별 영상을 일괄 NPZ 추출",
    "validate": "data/gestures/ NPZ 품질 검사 및 CSV 보고서 작성",
    "record": "웹캠 녹화와 일정 간격 이미지 추출",
    "flip": "two_fingers/ 이미지를 images/flipped/로 좌우 반전",
    "review": "이미지 PASS/trash 검토 Gradio UI 시작",
}


def check(argv: list[str]) -> int:
    from gesture import config
    parser = argparse.ArgumentParser(description="추론 자산 확인 (학습 데이터/카메라 접근 없음)")
    parser.add_argument("--model")
    parser.add_argument("--gestures")
    parser.add_argument("--static-model")
    parser.add_argument("--load-models", action="store_true", help="실제 모델 로드 및 metadata 호환성 확인")
    args = parser.parse_args(argv)
    paths = config.runtime_files(args.model, args.gestures, args.static_model)
    for name, path in paths.items():
        print(f"{'OK' if path.is_file() else 'MISSING'} {name}: {path}")
    if not all(p.is_file() for p in paths.values()):
        print("필요한 실제 모델 파일을 위 위치에 복사하세요. 학습 데이터는 추론에 필요하지 않습니다.")
        return 2
    if args.load_models:
        from gesture.model import GestureClassifier
        from gesture.landmarks import HandTracker
        GestureClassifier(model_file=paths["GRU/LSTM weight"])
        with HandTracker():
            pass
        if "static YOLO" in paths:
            from gesture_model import GestureDetector
            GestureDetector(paths["static YOLO"], device="cpu")
        print("모델 로드 및 metadata 확인 완료. 실제 카메라/키 입력은 별도 검증하세요.")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        parser = argparse.ArgumentParser(description="Jesture: 인자 없이 실행하면 통합 UI (연습 모드)",
                                         epilog="각 기능 도움말: python main.py <command> --help")
        parser.add_argument("command", nargs="?", choices=[*COMMANDS, "check"], default="app")
        parser.print_help()
        return 0
    command = argv.pop(0) if argv and not argv[0].startswith("-") else "app"
    previous_cwd, previous_argv = Path.cwd(), sys.argv
    try:
        # 모든 사용자 상대 경로는 저장소 루트 기준. 임의 CWD에서도 같은 자산 사용.
        os.chdir(ROOT)
        if command == "check":
            return check(argv)
        if command not in COMMANDS:
            raise ValueError(f"알 수 없는 command: {command}. python main.py --help를 확인하세요.")
        if command in INTERACTIVE_COMMANDS:
            argparse.ArgumentParser(prog=f"main.py {command}",
                                    description=INTERACTIVE_COMMANDS[command]).parse_args(argv)
        script = ROOT / COMMANDS[command]
        sys.argv = [str(script), *argv]
        runpy.run_path(str(script), run_name="__main__")
        return 0
    except (FileNotFoundError, ValueError, ImportError, OSError) as exc:
        print(f"실행 준비 오류: {exc}", file=sys.stderr)
        return 2
    finally:
        os.chdir(previous_cwd)
        sys.argv = previous_argv


if __name__ == "__main__":
    raise SystemExit(main())
