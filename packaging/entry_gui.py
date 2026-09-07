"""exe 진입점. 창 없는 빌드라 시작 실패를 로그 파일과 대화상자로 알립니다.

기존 `python -m service.gui`와 같은 인자를 그대로 받습니다. 실행 로직은 service.gui에 있고
여기서는 얼어붙은 실행에 필요한 준비와 오류 표시만 담당합니다.
"""

import multiprocessing
import os
from pathlib import Path
import sys
import traceback


def log_path():
    directory = os.environ.get("LOCALAPPDATA")
    if not directory:
        return Path(sys.executable).resolve().parent / "startup.log"
    return Path(directory) / "JestureService" / "startup.log"


def report(message):
    """콘솔이 없으므로 파일에 남기고 대화상자로 알립니다. 둘 다 실패해도 종료는 합니다."""
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(message, encoding="utf-8")
        message += f"\n\n자세한 내용: {path}"
    except OSError:
        pass
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, message, "Jesture 시작 실패", 0x10)
    except Exception:
        print(message, file=sys.stderr)


def main():
    # 얼어붙은 실행에서 자식 프로세스가 앱 전체를 다시 실행하지 않게 합니다.
    multiprocessing.freeze_support()
    try:
        from service.gui import main as run
    except Exception:
        report("필요한 구성 요소를 불러오지 못했습니다.\n\n" + traceback.format_exc())
        return 1
    try:
        return run()
    except SystemExit:
        raise
    except Exception:
        report("실행 중 오류가 발생했습니다.\n\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
