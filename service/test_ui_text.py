"""표시 번역과 내부 판정 계약이 분리되어 있는지 검사합니다."""

import ast
from pathlib import Path
import unittest

from service.ui_text import STATUS_NAMES, command_names, detail_text, gesture_name, model_line

SOURCES = ("gesture_controller.py", "hand_tracker.py")
# 이 함수들에 넘기는 고정 문자열도 화면에 그대로 나가는 사용자 문구입니다.
STATUS_CALLS = ("_invalidate_prediction", "missing_hand")


def has_hangul(text):
    return any("가" <= character <= "힣" for character in text)


def emitted_status_literals():
    """f-string이 아닌 고정 상태 문구만 모읍니다. 접두사 문구는 별도로 검사합니다."""
    found = set()
    for name in SOURCES:
        tree = ast.parse((Path(__file__).parent / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                    and any(isinstance(target, ast.Attribute) and target.attr == "reason"
                            for target in node.targets)):
                found.add(node.value.value)
            if not isinstance(node, ast.Call):
                continue
            called = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if called not in STATUS_CALLS:
                continue
            if not node.args:
                found.add("No hand")  # 기본 인자를 쓰는 missing_hand()도 같은 문구를 표시합니다.
            elif isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                found.add(node.args[0].value)
    return found


class TextTests(unittest.TestCase):
    def test_gesture_names(self):
        self.assertEqual(gesture_name("swipe_left"), "왼쪽으로 스와이프")
        self.assertEqual(gesture_name("make_fist"), "손 오므리기")
        self.assertEqual(gesture_name("finger_snap"), "핑거 스냅")

    def test_no_gesture_is_not_shown_to_users(self):
        labels = ("swipe_left", "make_fist", "no_gesture", "finger_snap")
        self.assertEqual(command_names(labels), ("왼쪽으로 스와이프", "손 오므리기", "핑거 스냅"))
        line = model_line("right-only", labels)
        self.assertEqual(line, "모델: 오른손 전용 · 왼쪽으로 스와이프, 손 오므리기, 핑거 스냅")
        for hidden in ("no_gesture", "right-only", "swipe_left"):
            self.assertNotIn(hidden, line)

    def test_model_line_before_and_without_finger_snap(self):
        self.assertEqual(model_line("", ()), "모델: 로딩 전")
        self.assertEqual(model_line("right-only", ("swipe_left", "make_fist", "no_gesture")),
                         "모델: 오른손 전용 · 왼쪽으로 스와이프, 손 오므리기")

    def test_every_known_status_reads_as_korean(self):
        for source in STATUS_NAMES:
            with self.subTest(source=source):
                self.assertTrue(has_hangul(detail_text(source)))
                self.assertNotIn(source, detail_text(source))

    def test_detail_pairs_and_prefixed_sources(self):
        self.assertEqual(detail_text("Right detected · Collecting fresh window"),
                         "오른손을 감지했습니다 · 동작을 확인하고 있습니다")
        self.assertIn("정보가 부족", detail_text("Quality hold: low_detection_rate;long_missing_run"))
        self.assertNotIn("low_detection_rate", detail_text("Quality hold: low_detection_rate"))
        self.assertEqual(detail_text("Expected Right, got Left (unsupported)"),
                         "왼손은 지원하지 않습니다. 오른손을 보여주세요")
        self.assertNotIn("Left", detail_text("Expected Right, got Left (unsupported)"))

    def test_same_advice_on_both_halves_is_shown_once(self):
        left = ("Expected Right, got Left (unsupported) · "
                "Left hand unsupported: show Right hand and return to neutral")
        self.assertEqual(detail_text(left), "왼손은 지원하지 않습니다. 오른손을 보여주세요")
        self.assertEqual(detail_text("No hand · No valid/allowed hand"),
                         "손이 보이지 않습니다 · 오른손을 카메라에 보여주세요")

    def test_translation_table_covers_every_emitted_status(self):
        emitted = emitted_status_literals()
        self.assertIn("No hand", emitted)
        self.assertIn("Collecting fresh window", emitted)
        missing = sorted(emitted - set(STATUS_NAMES))
        self.assertEqual(missing, [], f"ui_text.STATUS_NAMES에 번역이 없는 문구: {missing}")

    def test_unknown_sources_fall_back_without_leaking_internals(self):
        self.assertEqual(detail_text(""), "")
        self.assertTrue(has_hangul(detail_text("brand new internal reason")))
        self.assertNotIn("brand new", detail_text("brand new internal reason"))
        self.assertEqual(gesture_name("no_gesture"), "지원하지 않는 동작")


if __name__ == "__main__":
    unittest.main()
