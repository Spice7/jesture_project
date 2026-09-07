"""번들(exe)과 저장소 실행에서 같은 자산을 찾습니다. 파일을 열거나 만들지 않습니다.

PyInstaller는 `--add-data`로 넣은 파일을 `sys._MEIPASS` 아래 같은 상대 경로에 풉니다.
저장소에서 실행할 때는 프로젝트 루트를 기준으로 삼아 두 경우의 경로를 하나로 맞춥니다.
사용자 설정처럼 쓰기가 필요한 경로는 여기가 아니라 settings의 LOCALAPPDATA를 씁니다.
"""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def is_frozen():
    """PyInstaller로 묶인 실행 파일에서 동작 중인지 알려줍니다."""
    return bool(getattr(sys, "frozen", False)) and hasattr(sys, "_MEIPASS")


def resource_root():
    """읽기 전용 자산의 기준 폴더입니다. 번들에서는 풀린 임시/내부 폴더입니다."""
    if is_frozen():
        return Path(sys._MEIPASS)
    return REPOSITORY_ROOT


def resource_path(*parts):
    """`resource_path("models", "hand_landmarker.task")` 형태로 사용합니다."""
    if not parts or any(not isinstance(part, str) or not part for part in parts):
        raise ValueError("자산 경로 조각은 비어 있지 않은 문자열이어야 합니다.")
    return resource_root().joinpath(*parts)
