"""오른손 모델의 라벨 계약. 기존 번호를 유지하고 finger_snap만 추가합니다."""

LEGACY_RIGHT_LABELS = {"swipe_left": 0, "make_fist": 1, "no_gesture": 2}
RIGHT_LABELS = {**LEGACY_RIGHT_LABELS, "finger_snap": 3}


def label_map_for(schema):
    if schema != "right-only":
        raise ValueError("오른손 전용 모델만 지원합니다.")
    return dict(RIGHT_LABELS)


def validate_label_map(value):
    if (not isinstance(value, dict) or value not in (LEGACY_RIGHT_LABELS, RIGHT_LABELS)
            or any(type(v) is not int for v in value.values())):
        raise ValueError("오른손 라벨은 swipe_left=0, make_fist=1, no_gesture=2 및 선택적 finger_snap=3이어야 합니다.")


def schema_from_config(config):
    if not isinstance(config, dict):
        raise ValueError("전처리 config 객체가 필요합니다.")
    validate_label_map(config.get("label_map"))
    if (config.get("preprocessing_version") != "1.0.0"
            or config.get("augmentation") is not None
            or config.get("gesture_schema", "right-only") != "right-only"):
        raise ValueError("지원하지 않는 전처리 계약: 증강 없는 오른손 모델을 사용하세요.")
    return "right-only"


def validate_provenance(rows, config, report):
    """원본 파일 하나가 manifest 행 하나와 대응하는지 확인합니다."""
    schema_from_config(config)
    if report.get("discovered_files") != len(rows):
        raise ValueError("report와 manifest 파일 개수 불일치")
