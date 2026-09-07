"""오른손 전용 서비스의 손·명령 허용 목록입니다."""

SERVICE_HAND = "Right"
from training.gesture_schema import RIGHT_LABELS

SERVICE_LABELS = tuple(RIGHT_LABELS)
COMMAND_LABELS = tuple(label for label in SERVICE_LABELS if label != "no_gesture")


def supported_labels(model_labels):
    """모델 출력 번호를 재배열하지 않고 UI에서 사용할 라벨 이름만 추립니다."""
    return tuple(label for label in model_labels if label in SERVICE_LABELS)
