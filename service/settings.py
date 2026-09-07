"""사용자별 JSON 설정. 카메라/학습 자료와 분리하며 손상된 파일을 자동 덮어쓰지 않습니다."""

import copy
import json
import os
from pathlib import Path
import tempfile

from .resources import resource_path
from .shortcuts import Shortcut, OS_ACTION_CHORDS

DEFAULT_CHECKPOINT = resource_path("artifacts", "four_class_lstm_1layers_001", "best_model.pt")


def resolve_checkpoint(explicit=None, saved=""):
    """CLI 지정 > 저장된 사용자 경로 > 저장소 기준 기본 모델. 파일을 열지는 않습니다."""
    return Path(explicit or saved or DEFAULT_CHECKPOINT).expanduser().resolve()


def defaults():
    return {"version": 1, "shortcuts": {"swipe_left": Shortcut((), "Left").to_dict(),
            "make_fist": Shortcut((), "Space").to_dict(), "finger_snap": None},
            "toggle": Shortcut(("Ctrl", "Alt"), "G").to_dict(), "notifications": True,
            "preview": True, "camera_index": 0, "model_path": ""}


def validate_settings(value):
    if not isinstance(value, dict) or set(value) - set(defaults()):
        raise ValueError("지원하지 않는 설정 항목")
    result = defaults()
    result.update(copy.deepcopy(value))
    if type(result["version"]) is not int or result["version"] != 1:
        raise ValueError("지원하지 않는 설정 버전")
    if any(type(result[k]) is not bool for k in ("notifications", "preview")):
        raise ValueError("미리보기/알림 설정은 bool이어야 합니다.")
    if type(result["camera_index"]) is not int or not 0 <= result["camera_index"] <= 99:
        raise ValueError("카메라 번호는 0~99 정수여야 합니다.")
    if type(result["model_path"]) is not str or "\0" in result["model_path"]:
        raise ValueError("모델 경로 형식 오류")
    # 과거 설정 파일은 읽을 수 있도록 폐기된 항목만 메모리에서 제거합니다.
    # 로딩만으로 디스크의 사용자 설정을 덮어쓰지는 않습니다.
    if isinstance(result["shortcuts"], dict):
        result["shortcuts"].pop("swipe_right", None)
    if not isinstance(result["shortcuts"], dict) or set(result["shortcuts"]) - set(defaults()["shortcuts"]):
        raise ValueError("명령 라벨 설정 오류. no_gesture는 지정할 수 없습니다.")
    result["shortcuts"] = {**defaults()["shortcuts"], **result["shortcuts"]}
    toggle = Shortcut.from_dict(result["toggle"])
    if (frozenset(toggle.modifiers), toggle.key) in OS_ACTION_CHORDS:
        raise ValueError("Win+D와 Win+Shift+S는 제스처 실행용입니다. 전역 시작/중지에는 다른 조합을 지정하세요.")
    if not toggle.modifiers:
        raise ValueError("전역 토글에는 수정키가 필요합니다.")
    result["toggle"] = toggle.to_dict()
    for label, chord in result["shortcuts"].items():
        if chord is None:
            continue  # 미지정 명령은 인식 결과만 표시하고 키는 전송하지 않습니다.
        parsed = Shortcut.from_dict(chord)
        if parsed == toggle:
            raise ValueError(f"{label}: 전역 토글과 동일한 단축키는 사용할 수 없습니다.")
        result["shortcuts"][label] = parsed.to_dict()
    return result


def default_settings_path():
    directory = os.environ.get("LOCALAPPDATA")
    if not directory:
        raise RuntimeError("LOCALAPPDATA가 없습니다. --settings 경로를 지정하세요.")
    return Path(directory) / "JestureService" / "settings.json"


class SettingsStore:
    def __init__(self, path):
        self.path = Path(path)
        self.blocked = False
        self.loaded = False
        self.original = None

    def load(self):
        """호출 측은 실패를 표시하고 메모리 기본값으로만 동작할 수 있습니다."""
        try:
            self.original = self.path.read_bytes() if self.path.exists() else None
            def unique(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError(f"중복 설정 키: {key}")
                    result[key] = value
                return result
            result = validate_settings(json.loads(self.original.decode("utf-8-sig"), object_pairs_hook=unique)
                                       if self.original is not None else {})
        except (OSError, ValueError, UnicodeError, TypeError):
            self.blocked = True
            raise
        self.loaded, self.blocked = True, False
        return result

    def save(self, value):
        if self.blocked or not self.loaded:
            raise ValueError("설정 파일을 읽지 못했습니다. 기존 파일을 별도 보관/이동한 뒤 앱을 재시작하세요.")
        value = validate_settings(value)
        payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        current = self.path.read_bytes() if self.path.exists() else None
        if current != self.original:
            raise ValueError("다른 실행/프로그램에서 설정을 변경했습니다. 다시 실행하세요.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            descriptor, temporary = tempfile.mkstemp(prefix=".settings_", suffix=".tmp", dir=self.path.parent)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            self.original = payload
        finally:
            if temporary and Path(temporary).exists():
                Path(temporary).unlink()
