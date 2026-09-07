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
}

# 프리셋 이름 → {라벨: (keys, 표시 이름)}
PRESETS: dict[str, dict[str, tuple[list[str], str]]] = {
    "창 전환 (기본)": {
        "make_fist": (["ctrl", "alt", "tab"], "창 목록 열기"),
        "swipe_left": (["tab"], "다음 창"),
        "finger_snap": (["enter"], "창 선택"),
    },
    "발표 (PowerPoint)": {
        "finger_snap": (["f5"], "슬라이드쇼 시작"),
        "swipe_left": (["right"], "다음 슬라이드"),
        "make_fist": (["esc"], "슬라이드쇼 끝"),
    },
    "음악·영상": {
        "finger_snap": (["media_play_pause"], "재생/일시정지"),
        "swipe_left": (["media_next"], "다음 곡"),
        "make_fist": (["volume_mute"], "음소거"),
    },
    "브라우저": {
        "swipe_left": (["ctrl", "tab"], "다음 탭"),
        "make_fist": (["alt", "left"], "뒤로 가기"),
        "finger_snap": (["f5"], "새로고침"),
    },
    "바탕화면": {
        "make_fist": (["win", "d"], "바탕화면 보기"),
        "swipe_left": (["alt", "tab"], "직전 창"),
        "finger_snap": (["win", "tab"], "작업 보기"),
    },
}
