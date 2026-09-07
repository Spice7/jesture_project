"""제스처 → 키 매핑 프리셋. UI 의 '프리셋' 메뉴에서 한 번에 적용한다.

키 이름은 gesture.actions.VK 에 있는 이름을 쓴다. 사용자가 UI 에서 바꾼 매핑은 gestures.json 에 저장되므로
프리셋은 '출발점' 일 뿐이다.
"""
from __future__ import annotations

# 화면에 보여줄 제스처 이름 (라벨 → 한글, 짧은 설명)
GESTURE_NAMES = {
    "swipe_left": ("왼쪽 스와이프", "편 손을 자기 왼쪽으로 휙"),
    "make_fist": ("주먹 쥐기", "편 손을 쥐고 잠깐 멈춤"),
    "finger_snap": ("손가락 튕기기", "엄지+중지 붙였다 튕김"),
    "cancel": ("세 번째 포즈", "정적 포즈 유지 (YOLO)"),
}
# 정적 포즈(YOLO) 중 명령으로 쓰는 것. start/stop 은 게이트 열기/닫기라 매핑 대상이 아니다.
STATIC_LABELS = ["cancel"]

# 프리셋 이름 → {라벨: (keys, 표시 이름)}
PRESETS: dict[str, dict[str, tuple[list[str], str]]] = {
    "창 전환 (기본)": {
        "make_fist": (["ctrl", "alt", "tab"], "창 목록 열기"),
        "swipe_left": (["tab"], "다음 창"),
        "finger_snap": (["enter"], "창 선택"),
        "cancel": (["win", "d"], "바탕화면 보기"),
    },
    "발표 (PowerPoint)": {
        "finger_snap": (["f5"], "슬라이드쇼 시작"),
        "swipe_left": (["right"], "다음 슬라이드"),
        "make_fist": (["esc"], "슬라이드쇼 끝"),
        "cancel": (["b"], "화면 검게 (B)"),
    },
    "음악·영상": {
        "finger_snap": (["media_play_pause"], "재생/일시정지"),
        "swipe_left": (["media_next"], "다음 곡"),
        "make_fist": (["volume_mute"], "음소거"),
        "cancel": (["media_prev"], "이전 곡"),
    },
    "브라우저": {
        "swipe_left": (["ctrl", "tab"], "다음 탭"),
        "make_fist": (["alt", "left"], "뒤로 가기"),
        "finger_snap": (["f5"], "새로고침"),
        "cancel": (["ctrl", "t"], "새 탭"),
    },
    "바탕화면": {
        "make_fist": (["win", "d"], "바탕화면 보기"),
        "swipe_left": (["alt", "tab"], "직전 창"),
        "finger_snap": (["win", "tab"], "작업 보기"),
        "cancel": (["win", "e"], "탐색기 열기"),
    },
}

# "기능 고르기" 목록: (분류, 이름, keys). 키 조합을 모르는 사용자가 이름으로 고르게 (Logi Options+ 방식)
FUNCTIONS: list[tuple[str, str, list[str]]] = [
    ("창", "창 목록 열기", ["ctrl", "alt", "tab"]),
    ("창", "다음 창 (목록 안)", ["tab"]),
    ("창", "창 선택 (목록 안)", ["enter"]),
    ("창", "직전 창", ["alt", "tab"]),
    ("창", "작업 보기", ["win", "tab"]),
    ("창", "바탕화면 보기", ["win", "d"]),
    ("창", "탐색기 열기", ["win", "e"]),
    ("창", "창 닫기", ["alt", "f4"]),
    ("미디어", "재생/일시정지", ["media_play_pause"]),
    ("미디어", "다음 곡", ["media_next"]),
    ("미디어", "이전 곡", ["media_prev"]),
    ("미디어", "소리 크게", ["volume_up"]),
    ("미디어", "소리 작게", ["volume_down"]),
    ("미디어", "음소거", ["volume_mute"]),
    ("발표", "슬라이드쇼 시작", ["f5"]),
    ("발표", "다음 슬라이드", ["right"]),
    ("발표", "이전 슬라이드", ["left"]),
    ("발표", "슬라이드쇼 끝", ["esc"]),
    ("발표", "화면 검게", ["b"]),
    ("브라우저", "새 탭", ["ctrl", "t"]),
    ("브라우저", "다음 탭", ["ctrl", "tab"]),
    ("브라우저", "이전 탭", ["ctrl", "shift", "tab"]),
    ("브라우저", "탭 닫기", ["ctrl", "w"]),
    ("브라우저", "뒤로", ["alt", "left"]),
    ("브라우저", "앞으로", ["alt", "right"]),
    ("브라우저", "새로고침", ["f5"]),
    ("편집", "복사", ["ctrl", "c"]),
    ("편집", "붙여넣기", ["ctrl", "v"]),
    ("편집", "실행 취소", ["ctrl", "z"]),
    ("기타", "화면 캡처", ["printscreen"]),
    ("기타", "캡처 도구", ["win", "shift", "s"]),
]
