"""CPU 전용 빌드 환경을 만들고 Jesture GUI를 exe로 묶습니다.

개발/학습용 `.venv`는 건드리지 않습니다. CUDA torch(4GB)가 결과물에 들어가지 않도록
`.venv-build`에 CPU torch를 따로 설치하고 그 환경에서 PyInstaller를 실행합니다.

    .\\.venv\\Scripts\\python.exe .\\packaging\\build_exe.py

결과는 `dist/Jesture/Jesture.exe`입니다. 폴더 전체가 하나의 프로그램이므로
exe만 따로 옮기면 실행되지 않습니다.
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BUILD_VENV = ROOT / ".venv-build"
SPEC = ROOT / "packaging" / "jesture.spec"
# 서비스 실행에 실제로 필요한 것만 설치합니다. tensorflow/matplotlib/pandas는 쓰지 않습니다.
CPU_REQUIREMENTS = [
    "torch==2.14.0", "torchvision==0.29.0",
    "--index-url", "https://download.pytorch.org/whl/cpu",
]
REQUIREMENTS = [
    "pyinstaller>=6.11",
    "mediapipe>=1.0.1", "opencv-python>=5.0.0.93", "numpy>=2.5.2",
    "ultralytics>=8.4.138", "PySide6>=6.8,<7",
]


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recreate", action="store_true",
                        help="빌드 환경을 지우고 처음부터 다시 만듭니다.")
    parser.add_argument("--skip-install", action="store_true",
                        help="이미 준비된 빌드 환경을 그대로 쓰고 설치 단계를 건너뜁니다.")
    parser.add_argument("--keep-build", action="store_true",
                        help="PyInstaller의 중간 산출물(build/)을 지우지 않습니다.")
    return parser


def run(command, **kwargs):
    print("+", " ".join(str(part) for part in command), flush=True)
    result = subprocess.run(command, cwd=ROOT, **kwargs)
    if result.returncode != 0:
        raise SystemExit(f"명령이 실패했습니다 (코드 {result.returncode})")


def build_python():
    return BUILD_VENV / "Scripts" / "python.exe"


def ensure_environment(recreate, skip_install):
    if recreate and BUILD_VENV.exists():
        print(f"기존 빌드 환경 제거: {BUILD_VENV}", flush=True)
        shutil.rmtree(BUILD_VENV)
    if skip_install:
        if not build_python().is_file():
            raise SystemExit(f"빌드 환경이 없습니다. --skip-install 없이 먼저 실행하세요: {BUILD_VENV}")
        return
    if not build_python().is_file():
        run([sys.executable, "-m", "venv", str(BUILD_VENV)])
    python = build_python()
    run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    # CPU 휠을 먼저 고정해야 다른 패키지가 CUDA torch를 끌어오지 않습니다.
    run([str(python), "-m", "pip", "install", *CPU_REQUIREMENTS])
    run([str(python), "-m", "pip", "install", *REQUIREMENTS])
    verify(python)


def verify(python):
    """묶기 전에 빌드 환경이 CPU torch인지, 자산을 찾는지 확인합니다."""
    script = (
        "import torch, sys;"
        "print('torch', torch.__version__);"
        "sys.exit('CUDA 빌드가 설치되었습니다. --recreate로 다시 만드세요.')"
        " if '+cu' in torch.__version__ else None"
    )
    run([str(python), "-c", script])


def main(argv=None):
    args = make_parser().parse_args(argv)
    if sys.platform != "win32":
        raise SystemExit("이 빌드 스크립트는 Windows 전용입니다.")
    if not SPEC.is_file():
        raise SystemExit(f"스펙 파일이 없습니다: {SPEC}")
    ensure_environment(args.recreate, args.skip_install)
    distribution = ROOT / "dist" / "Jesture"
    if distribution.exists():
        print(f"이전 결과 제거: {distribution}", flush=True)
        shutil.rmtree(distribution)
    # -X utf8이 없으면 스펙이 내는 한국어 오류가 콘솔 코드페이지에서 깨집니다.
    run([str(build_python()), "-X", "utf8", "-m", "PyInstaller", "--noconfirm", "--clean", str(SPEC)])
    executable = distribution / "Jesture.exe"
    if not executable.is_file():
        raise SystemExit("빌드는 끝났지만 실행 파일을 찾지 못했습니다.")
    if not args.keep_build:
        shutil.rmtree(ROOT / "build", ignore_errors=True)
    total = sum(path.stat().st_size for path in distribution.rglob("*") if path.is_file())
    print(f"\n완료: {executable}")
    print(f"배포 폴더 크기: {total/1024/1024/1024:.2f} GB")
    print("폴더 전체를 함께 전달하세요. exe만 옮기면 실행되지 않습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
