"""확정 이벤트만 로그로 표시합니다. 실제 키 입력, 외부 앱 제어 및 파일 저장은 없습니다."""

ACTION_NAMES = {"swipe_left": "SWIPE_LEFT_ACTION", "make_fist": "MAKE_FIST_ACTION",
                "finger_snap": "FINGER_SNAP_ACTION"}


def dispatch(gesture):
    """추후 단축키 실행을 연결할 경계입니다. no_gesture에는 아무 일도 하지 않습니다."""
    if gesture == "no_gesture":
        return
    if gesture not in ACTION_NAMES:
        raise ValueError(f"알 수 없는 제스처 이벤트: {gesture}")
    print(f"[DRY RUN] gesture={gesture}, action={ACTION_NAMES[gesture]}")
