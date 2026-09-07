"""Windows GUI 진입점. --help와 import에는 PySide6 및 장치 초기화가 필요하지 않습니다."""

import argparse
from pathlib import Path
import sys


def make_parser():
    parser = argparse.ArgumentParser(description="제스처 GUI: 실행 시 OFF, 사용자가 시작하면 실제 단축키를 전송합니다.")
    parser.add_argument("--checkpoint", type=Path, help="이번 실행의 모델. 생략 시 저장 경로, 없으면 lstm_gpu_2layers_001 사용")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--settings", type=Path, help="설정 경로 재정의. 기본 LOCALAPPDATA/JestureService/settings.json")
    parser.add_argument("--diagnostics", type=Path, help="선택적 진단 JSONL 경로. 기존 파일은 덮어쓰지 않습니다.")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        from .gui_widgets import run_gui
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("PySide6"):
            print("PySide6가 없습니다. 프로젝트 가상 환경에 PySide6를 설치한 뒤 다시 실행하세요. service/README.md 참고.",
                  file=sys.stderr)
            return 2
        raise
    if sys.platform != "win32":
        print("이 GUI 버전은 Windows 전용입니다.", file=sys.stderr)
        return 2
    try:
        return run_gui(args)
    except Exception as exc:
        print(f"GUI 시작 실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
