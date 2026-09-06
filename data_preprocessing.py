"""Gradio app for quickly reviewing and cleaning an image folder."""

from __future__ import annotations

import json
import re
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    import gradio as gr
    from PIL import Image, ImageOps
except ImportError as exc:  # Give a useful message when the file is run directly.
    raise SystemExit(
        "Gradio가 설치되어 있지 않습니다. `pip install gradio` 실행 후 다시 시작하세요."
    ) from exc


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
}
TRASH_DIR_NAME = ".trash"
HISTORY_FILE_NAME = ".data_preprocessing_passed.json"


def natural_sort_key(path: Path) -> list[tuple[int, object]]:
    """Sort frame_2 before frame_10."""
    return [
        (1, int(part)) if part.isdigit() else (0, part.lower())
        for part in re.split(r"(\d+)", path.as_posix())
    ]


@dataclass
class ReviewSession:
    root: Path | None = None
    images: list[Path] = field(default_factory=list)
    session_total: int = 0
    reviewed: int = 0
    passed_paths: set[str] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def _history_path(self) -> Path:
        assert self.root is not None
        return self.root / HISTORY_FILE_NAME

    def _load_history(self) -> None:
        self.passed_paths = set()
        history_path = self._history_path()
        if not history_path.exists():
            return

        try:
            data = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self.passed_paths = {str(item) for item in data}
        except (OSError, json.JSONDecodeError):
            # A broken history file should not prevent image review.
            self.passed_paths = set()

    def _save_history(self) -> None:
        history_path = self._history_path()
        temp_path = history_path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(sorted(self.passed_paths), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(history_path)

    def load(self, folder: str) -> tuple[object, str, str, str]:
        with self.lock:
            raw_folder = (folder or "").strip().strip('"')
            if not raw_folder:
                return None, "이미지 없음", "0 / 0", "폴더 경로를 입력하세요."

            root = Path(raw_folder).expanduser().resolve()
            if not root.is_dir():
                return None, "이미지 없음", "0 / 0", f"폴더를 찾을 수 없습니다: {root}"

            self.root = root
            self._load_history()

            all_images = [
                path
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
                and TRASH_DIR_NAME not in path.relative_to(root).parts
            ]
            all_images.sort(key=lambda path: natural_sort_key(path.relative_to(root)))

            self.images = [
                path
                for path in all_images
                if path.relative_to(root).as_posix() not in self.passed_paths
            ]
            self.session_total = len(self.images)
            self.reviewed = 0

            skipped = len(all_images) - len(self.images)
            status = f"이미지 {len(all_images)}장을 찾았습니다."
            if skipped:
                status += f" 이전에 pass한 {skipped}장은 건너뜁니다."
            return self.current_view(status)

    def current_view(self, status: str = "") -> tuple[object, str, str, str]:
        if not self.images or self.root is None:
            final_status = status or "모든 이미지 검토가 끝났습니다."
            return None, "이미지 없음", f"{self.reviewed} / {self.session_total}", final_status

        current = self.images[0]
        relative = current.relative_to(self.root).as_posix()
        try:
            # Copying closes the source file immediately, so Windows can move it later.
            with Image.open(current) as source:
                image = ImageOps.exif_transpose(source).copy()
        except (OSError, ValueError) as exc:
            return (
                None,
                relative,
                f"{self.reviewed} / {self.session_total}",
                f"이미지를 열 수 없습니다: {exc}. trash로 옮기거나 폴더를 다시 불러오세요.",
            )

        return image, relative, f"{self.reviewed} / {self.session_total}", status

    def pass_current(self) -> tuple[object, str, str, str]:
        with self.lock:
            if not self.images or self.root is None:
                return self.current_view("처리할 이미지가 없습니다. 먼저 폴더를 불러오세요.")

            current = self.images.pop(0)
            relative = current.relative_to(self.root).as_posix()
            self.passed_paths.add(relative)
            try:
                self._save_history()
            except OSError as exc:
                # Do not lose the item when the decision could not be recorded.
                self.images.insert(0, current)
                self.passed_paths.discard(relative)
                return self.current_view(f"진행 기록 저장 실패: {exc}")

            self.reviewed += 1
            return self.current_view(f"PASS: {relative}")

    def trash_current(self) -> tuple[object, str, str, str]:
        with self.lock:
            if not self.images or self.root is None:
                return self.current_view("처리할 이미지가 없습니다. 먼저 폴더를 불러오세요.")

            current = self.images[0]
            relative_path = current.relative_to(self.root)
            destination = self.root / TRASH_DIR_NAME / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)

            if destination.exists():
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                destination = destination.with_name(
                    f"{destination.stem}_{stamp}{destination.suffix}"
                )

            try:
                shutil.move(str(current), str(destination))
            except OSError as exc:
                return self.current_view(f"파일 이동 실패: {exc}")

            self.images.pop(0)
            self.reviewed += 1
            return self.current_view(f"TRASH: {relative_path.as_posix()}")


session = ReviewSession()


def build_app() -> gr.Blocks:
    with gr.Blocks(title="이미지 데이터 정제") as app:
        gr.Markdown("# 이미지 데이터 정제")
        gr.Markdown(
            "폴더를 불러온 뒤 **pass**는 유지, **trash**는 폴더 안의 `.trash`로 이동합니다."
        )

        with gr.Row():
            folder = gr.Textbox(
                label="이미지 폴더",
                value="./frames",
                placeholder=r"예: C:\data\images 또는 ./frames",
                scale=5,
            )
            load_button = gr.Button("폴더 불러오기", variant="primary", scale=1)

        image = gr.Image(label="현재 이미지", type="pil", height=650)

        with gr.Row():
            file_name = gr.Textbox(label="파일", interactive=False)
            progress = gr.Textbox(label="진행", interactive=False)

        status = gr.Textbox(label="상태", interactive=False)

        with gr.Row():
            pass_button = gr.Button("PASS", variant="primary", size="lg")
            trash_button = gr.Button("TRASH", variant="stop", size="lg")

        outputs = [image, file_name, progress, status]
        load_button.click(session.load, inputs=folder, outputs=outputs)
        folder.submit(session.load, inputs=folder, outputs=outputs)
        pass_button.click(session.pass_current, outputs=outputs)
        trash_button.click(session.trash_current, outputs=outputs)

    return app


if __name__ == "__main__":
    build_app().launch(inbrowser=True)
