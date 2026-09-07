"""번들 경로 해석과 스펙의 자산 배치가 서로 맞는지 검사합니다. exe를 만들지는 않습니다."""

from pathlib import Path
import sys
import unittest
from unittest import mock

from service.hand_tracker import DEFAULT_MODEL
from service.resources import REPOSITORY_ROOT, is_frozen, resource_path, resource_root
from service.settings import DEFAULT_CHECKPOINT
from service.yolo_gate import DEFAULT_WEIGHTS

SPEC = REPOSITORY_ROOT / "packaging" / "jesture.spec"
BUNDLED = (DEFAULT_MODEL, DEFAULT_WEIGHTS, DEFAULT_CHECKPOINT)


class ResourceTests(unittest.TestCase):
    def test_repository_run_uses_project_root(self):
        self.assertFalse(is_frozen())
        self.assertEqual(resource_root(), REPOSITORY_ROOT)
        self.assertEqual(resource_path("models", "a.task"), REPOSITORY_ROOT / "models" / "a.task")

    def test_frozen_run_uses_the_unpacked_bundle(self):
        unpacked = Path(__file__).resolve().parent
        with mock.patch.object(sys, "frozen", True, create=True), \
             mock.patch.object(sys, "_MEIPASS", str(unpacked), create=True):
            self.assertTrue(is_frozen())
            self.assertEqual(resource_root(), unpacked)
            self.assertEqual(resource_path("yolo", "v2", "best.pt"), unpacked / "yolo" / "v2" / "best.pt")

    def test_rejects_empty_or_non_string_parts(self):
        for bad in ((), ("",), ("models", None), (Path("models"),)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    resource_path(*bad)


@unittest.skipUnless(SPEC.is_file(), f"스펙 파일 없음: {SPEC}")
class SpecTests(unittest.TestCase):
    """스펙이 자산을 코드가 찾는 상대 경로에 넣는지 확인합니다."""

    def setUp(self):
        self.spec = SPEC.read_text(encoding="utf-8")

    def test_every_default_asset_is_listed_in_the_spec(self):
        for path in BUNDLED:
            relative = path.relative_to(REPOSITORY_ROOT)
            with self.subTest(asset=str(relative)):
                self.assertTrue(path.is_file(), f"자산 파일이 없습니다: {path}")
                # 스펙은 파일 이름과 그 상위 폴더를 번들 대상 경로로 함께 적습니다.
                self.assertIn(f'"{path.name}"', self.spec)
                target = relative.parent.as_posix()
                self.assertIn(f'"{target}"', self.spec)

    def test_unused_heavy_dependencies_stay_excluded(self):
        for name in ("tensorflow", "keras", "ipykernel", "tkinter"):
            with self.subTest(name=name):
                self.assertIn(f'"{name}"', self.spec)

    def test_standard_library_and_torch_internals_are_not_excluded(self):
        """torch가 unittest를 import합니다. 표준 라이브러리 제외는 실행 시점에야 터집니다."""
        excludes = self.spec.split("excludes = [", 1)[1].split("]", 1)[0]
        for name in ("unittest", "pydoc", "doctest", "torch.", "scipy."):
            with self.subTest(name=name):
                self.assertNotIn(f'"{name}', excludes)

    def test_torchvision_extension_is_bundled_explicitly(self):
        """확장이 빠지면 로딩은 되고 YOLO의 NMS 단계에서만 죽습니다."""
        self.assertIn("torchvision", self.spec)
        self.assertIn('glob("*.pyd")', self.spec)
        self.assertIn("torchvision::nms", self.spec)

    def test_lazy_imports_are_declared_as_hidden(self):
        # predictor가 전처리 계약 검증에 training 패키지를 그대로 씁니다.
        for name in ("training.prepare_dataset", "training.gesture_schema"):
            with self.subTest(name=name):
                self.assertIn(name, self.spec)
        for name in ("mediapipe", "ultralytics"):
            with self.subTest(name=name):
                self.assertIn(f'collect_data_files("{name}")', self.spec)


if __name__ == "__main__":
    unittest.main()
