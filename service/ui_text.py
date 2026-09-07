"""사용자 표시용 한국어. 모델 라벨과 진단 로그의 원문은 바꾸지 않습니다."""

GESTURE_NAMES = {
    "swipe_left": "왼쪽으로 스와이프",
    "make_fist": "손 오므리기",
    "finger_snap": "핑거 스냅",
}
# no_gesture는 중립 판정에만 쓰는 내부 라벨이므로 사용자 화면에 노출하지 않습니다.
COMMAND_ORDER = ("swipe_left", "make_fist", "finger_snap")

STATUS_NAMES = {
    # hand_tracker의 손 상태
    "No hand": "손이 보이지 않습니다",
    "Right detected": "오른손을 감지했습니다",
    "Multiple hands unsupported": "오른손 하나만 보여주세요",
    "Unknown handedness": "손 방향을 확인할 수 없습니다",
    "Invalid landmarks": "손 위치를 정확히 확인할 수 없습니다",
    # gesture_controller의 보류/진행 사유
    "Recognition OFF": "인식이 중지되었습니다",
    "Collecting fresh window": "동작을 확인하고 있습니다",
    "No valid/allowed hand": "오른손을 카메라에 보여주세요",
    "Low confidence": "동작이 불확실해 실행을 보류합니다",
    "Neutral / no command": "다음 동작을 기다리고 있습니다",
    "Stabilizing prediction": "동작을 한 번 더 확인하고 있습니다",
    "Wait for neutral before next command": "다음 명령을 위해 잠시 자연스러운 손 자세로 돌아와 주세요",
    "Event confirmed (DRY RUN)": "동작을 확인했습니다",
    "Hand/class mismatch: command blocked": "지원하는 손동작이 아니어서 실행을 보류합니다",
    "Input latency/gap: fresh window and neutral required": "입력이 끊기거나 화면이 바뀌어 동작을 다시 확인합니다",
    "Camera time gap: fresh window required": "카메라 입력이 지연되어 동작을 다시 확인합니다",
    "Hand changed: fresh window and neutral required": "손이 바뀌어 동작을 다시 확인합니다",
    "Hand lost: buffer cleared, wait for neutral": "손을 놓쳤습니다. 오른손을 보여주고 잠시 자연스러운 자세로 기다려주세요",
    "Left hand unsupported: show Right hand and return to neutral": "왼손은 지원하지 않습니다. 오른손을 보여주세요",
    "Input or inference error; recognition OFF": "입력 처리에 문제가 있어 인식을 중지했습니다",
}


def gesture_name(label):
    """명령 라벨의 한국어 이름입니다. 표시 전용이며 판정에는 쓰지 않습니다."""
    return GESTURE_NAMES.get(label, "지원하지 않는 동작")


def command_names(labels):
    """모델이 지원하는 명령만 정해진 순서로 돌려줍니다. no_gesture는 제외합니다."""
    return tuple(GESTURE_NAMES[name] for name in COMMAND_ORDER if name in labels)


def model_line(schema, labels):
    """메인 화면의 모델 요약 한 줄입니다. 로딩 전에는 명령 목록을 감춥니다."""
    commands = command_names(labels)
    text = "모델: " + ("오른손 전용" if schema else "로딩 전")
    return text + (" · " + ", ".join(commands) if commands else "")


def detail_text(detail):
    """`손 상태 · 보류 사유` 형태의 내부 문구를 사용자 문장으로 바꿉니다."""
    if not detail:
        return ""
    shown = []
    for part in detail.split(" · "):
        sentence = translate_status(part.strip())
        # 손 상태와 보류 사유가 같은 안내로 번역되면 한 번만 보여줍니다.
        if sentence not in shown:
            shown.append(sentence)
    return " · ".join(shown)


def translate_status(part):
    # 손 이름이나 품질 원인 토큰까지 그대로 보여주지 않고 할 일만 안내합니다.
    if part.startswith("Expected Right, got "):
        return STATUS_NAMES["Left hand unsupported: show Right hand and return to neutral"]
    if part.startswith("Quality hold:"):
        return "손동작 정보가 부족합니다. 손 전체가 잘 보이도록 해주세요"
    return STATUS_NAMES.get(part, "상태를 확인하고 있습니다")
