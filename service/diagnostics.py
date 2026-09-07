"""선택적 JSONL 진단. 영상/좌표/키 내용 없이 시간과 분류 판정만 저장합니다."""

from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import sys
import threading
import time


class DiagnosticLog:
    """유한 큐의 별도 작업자가 저장합니다. 로그 실패는 인식 조건에 영향을 주지 않습니다."""

    def __init__(self, path, max_bytes=20_000_000):
        self.path = Path(path)
        self.stream = self.path.open("x", encoding="utf-8")  # 기존 로그 덮어쓰기 금지
        self.origin = time.monotonic()
        self.queue = queue.Queue(maxsize=4096)
        self.stopping = threading.Event()
        self.failed = False
        self.dropped = 0
        self.max_bytes = max_bytes
        self.thread = threading.Thread(target=self._write, name="gesture-diagnostics", daemon=True)
        self.thread.start()

    def emit(self, event, **fields):
        if self.failed or self.stopping.is_set():
            return
        now = time.monotonic()
        row = dict(event=event, monotonic_s=now, elapsed_s=now-self.origin,
                   utc=datetime.now(timezone.utc).isoformat(), **fields)
        try:
            self.queue.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def _write(self):
        written = 0
        try:
            while not self.stopping.is_set() or not self.queue.empty():
                try:
                    row = self.queue.get(timeout=.1)
                except queue.Empty:
                    continue
                payload = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                written += len(payload.encode("utf-8"))
                if written > self.max_bytes:
                    raise OSError("진단 로그 크기 상한 도달")
                self.stream.write(payload)
                self.stream.flush()
            self.stream.write(json.dumps({"event": "diagnostics_closed", "dropped": self.dropped}) + "\n")
        except Exception as exc:
            self.failed = True
            print(f"진단 로그 저장 중단 (인식은 유지): {exc}", file=sys.stderr)
        finally:
            self.stream.close()

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=2)
