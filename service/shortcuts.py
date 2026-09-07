"""조합키의 자료 계약과 안전한 전송. GUI/Windows API 없이도 검증할 수 있습니다."""

from dataclasses import dataclass

MODIFIERS = ("Ctrl", "Alt", "Shift", "Win")
KEYS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") + tuple(f"F{i}" for i in range(1, 13)) + (
    "Left", "Right", "Up", "Down", "Space", "Enter", "Tab", "Backspace", "Delete",
    "Insert", "Home", "End", "PageUp", "PageDown")

# 사용자가 요청한 OS 동작은 제스처의 출력으로만 허용합니다.
# 다른 수정키가 더 붙은 조합까지 포괄적으로 허용하지 않습니다.
OS_ACTION_CHORDS = {(frozenset({"Win"}), "D"), (frozenset({"Win", "Shift"}), "S")}


@dataclass(frozen=True)
class Shortcut:
    """수정키 목록 + 일반 키 하나. 표시 문자열을 코드나 명령으로 실행하지 않습니다."""

    modifiers: tuple[str, ...]
    key: str

    def __post_init__(self):
        if (not isinstance(self.modifiers, tuple) or any(m not in MODIFIERS for m in self.modifiers)
                or len(set(self.modifiers)) != len(self.modifiers) or self.key not in KEYS):
            raise ValueError("지원하는 수정키와 일반 키 하나가 필요합니다. Esc는 취소 전용입니다.")
        object.__setattr__(self, "modifiers", tuple(m for m in MODIFIERS if m in self.modifiers))
        mods = set(self.modifiers)
        # 모든 OS/앱 예약 키를 열거할 수는 없습니다. 명백한 시스템 조합은 미리 거부합니다.
        reserved_win = set("LDERISAXGVMUPKH WZCNTBQ".replace(" ", "")) | {
            "Tab", "Space", "Up", "Down", "Left", "Right", "Enter", "Home", "PageUp", "PageDown"}
        allowed_os_action = (frozenset(mods), self.key) in OS_ACTION_CHORDS
        if (("Win" in mods and self.key in reserved_win and not allowed_os_action)
                or (self.key == "Delete" and {"Ctrl", "Alt"} <= mods)
                or ("Alt" in mods and self.key in ("Tab", "F4", "Space"))
                or ("Ctrl" in mods and self.key == "F4") or self.key == "F12"):
            raise ValueError("시스템 예약/안전 정책으로 지원하지 않는 조합입니다. F12도 제외합니다.")

    def text(self):
        return "+".join((*self.modifiers, self.key))

    def to_dict(self):
        return {"modifiers": list(self.modifiers), "key": self.key}

    @classmethod
    def from_dict(cls, value):
        if (not isinstance(value, dict) or set(value) != {"modifiers", "key"}
                or not isinstance(value["modifiers"], list)
                or any(type(m) is not str for m in value["modifiers"])
                or type(value["key"]) is not str):
            raise ValueError("단축키 설정 형식 오류")
        return cls(tuple(value["modifiers"]), value["key"])


class ChordCapture:
    """설정 위젯 안의 키 이벤트만 처리합니다. 전역 타이핑을 수집하지 않습니다."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.held = set()
        self.value = None
        self.cancelled = False

    def press(self, key, repeat=False):
        if repeat or self.cancelled:
            return
        if key == "Esc":
            self.reset()
            self.cancelled = True
        elif key in MODIFIERS:
            self.held.add(key)
        else:
            # 새 입력이 잘못되어도 이전 조합을 저장하게 두지 않습니다.
            self.value = None
            if self.held - set(MODIFIERS):
                raise ValueError("일반 키는 한 번에 하나만 지원합니다. 키를 모두 떼고 다시 입력하세요.")
            self.held.add(key)
            self.value = Shortcut(tuple(m for m in MODIFIERS if m in self.held), key)

    def release(self, key):
        self.held.discard(key)


class KeySender:
    """현재 대상/물리 키 상태를 검사한 뒤 한 번만 전송합니다. 실패한 명령은 재시도하지 않습니다."""

    def __init__(self, backend):
        self.backend = backend
        self.owned = []

    def send(self, shortcut, target):
        # foreground는 (창 핸들, PID). 본 앱과 불명확한 대상은 항상 차단합니다.
        if (not target or not target[0] or not target[1] or target[1] == self.backend.process_id
                or self.backend.foreground() != target):
            return False, "대상 창 변경/본 앱 활성화: 입력 생략"
        keys = [*shortcut.modifiers, shortcut.key]
        if self.backend.any_pressed(tuple(dict.fromkeys((*MODIFIERS, *keys)))):
            return False, "사용자가 키를 누르고 있어 입력 생략"
        if self.owned:
            raise RuntimeError("이전 키 해제 실패: 입력을 중지하고 앱을 종료하세요.")
        events = [(key, True) for key in keys] + [(key, False) for key in reversed(keys)]
        # 한 SendInput 배치에 넣어 누르기/떼기 사이의 지연을 최소화합니다.
        if self.backend.foreground() != target:
            return False, "대상 창 변경: 입력 생략"
        count = self.backend.send_events(events)
        if type(count) is not int or not 0 <= count <= len(events):
            raise RuntimeError("키 전송 반환 계약 오류")
        for key, down in events[:count]:
            if down:
                self.owned.append(key)
            else:
                self.owned.remove(key)
        if count != len(events):
            self.release_owned()
            raise RuntimeError("키 입력 일부/전체 전송 실패. 권한 및 대상 앱을 확인하세요. 자동 재시도하지 않습니다.")
        return True, f"{shortcut.text()} 입력 전송"

    def release_owned(self):
        """전송 API가 성공했다고 보고한 본 프로그램의 down만 해제합니다."""
        errors = []
        for key in reversed(self.owned.copy()):
            try:
                if self.backend.send_events([(key, False)]) != 1:
                    raise RuntimeError(f"{key} 해제 실패")
                self.owned.remove(key)
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("; ".join(errors))


class HotkeyManager:
    """등록 교체와 실패 시 복원. backend는 반드시 같은 GUI 스레드에서 호출합니다."""

    def __init__(self, backend):
        self.backend = backend
        self.current = None
        self.active_id = None
        self.next_id = 1
        self.registrations = {}

    def _register(self, identifier, shortcut):
        if (frozenset(shortcut.modifiers), shortcut.key) in OS_ACTION_CHORDS:
            raise ValueError("Win+D와 Win+Shift+S는 제스처 실행용입니다. 전역 시작/중지에는 다른 조합을 지정하세요.")
        self.backend.register_hotkey(identifier, shortcut)
        self.registrations[identifier] = shortcut

    def _unregister(self, identifier):
        self.backend.unregister_hotkey(identifier)
        self.registrations.pop(identifier, None)

    def replace(self, shortcut, persist=lambda: None):
        if not shortcut.modifiers:
            raise ValueError("전역 제어에는 Ctrl/Alt/Shift/Win 중 하나 이상의 수정키가 필요합니다.")
        if shortcut == self.current:
            persist()
            return
        new_id = self.next_id
        self.next_id = self.next_id % 0xBFFF + 1
        self._register(new_id, shortcut)
        old_id, old = self.active_id, self.current
        try:
            if old_id is not None:
                self._unregister(old_id)
        except Exception:
            self._unregister(new_id)
            raise
        try:
            persist()
        except Exception as error:
            self.active_id = self.current = None
            try:
                self._unregister(new_id)
                if old_id is not None:
                    self._register(old_id, old)
                    self.active_id, self.current = old_id, old
            except Exception as restore:
                raise RuntimeError(f"설정 저장 실패 및 전역 단축키 복원 실패: {restore}") from error
            raise
        self.active_id, self.current = new_id, shortcut

    def close(self):
        # 교체/복원 도중 실패한 등록도 추적하여 가능한 항목은 전부 정리합니다.
        errors = []
        for identifier in list(self.registrations):
            try:
                self._unregister(identifier)
            except Exception as exc:
                errors.append(str(exc))
        if self.active_id not in self.registrations:
            self.active_id = self.current = None
        if errors:
            raise RuntimeError("전역 단축키 해제 실패: " + "; ".join(errors))
