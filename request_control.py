"""Cancellation tokens shared by ASR and retrieval stages in one API process."""
from contextlib import contextmanager
import threading
import time
import uuid

from asr_worker import ASRServiceError


def check_cancelled(event):
    if event is not None and event.is_set():
        raise ASRServiceError(499, "REQUEST_CANCELLED", "요청을 취소했습니다.")


class RequestRegistry:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = {}
        self.cancelled = {}

    def _prune(self):
        now = time.monotonic()
        self.cancelled = {key: expiry for key, expiry in self.cancelled.items() if expiry > now}

    @contextmanager
    def track(self, request_id=None):
        request_id = request_id or str(uuid.uuid4())
        event = threading.Event()
        with self.lock:
            self._prune()
            if request_id in self.cancelled:
                event.set()
            check_cancelled(event)
            if request_id in self.active:
                raise ASRServiceError(409, "DUPLICATE_REQUEST", "요청 ID가 이미 실행 중입니다.")
            self.active[request_id] = event
        try:
            yield event
        finally:
            with self.lock:
                self.active.pop(request_id, None)

    def cancel(self, request_id):
        with self.lock:
            self._prune()
            event = self.active.get(request_id)
            if event is not None:
                event.set()
            # Handle cancellation arriving before upload/decode or between stages.
            self.cancelled[request_id] = time.monotonic() + 300
            while len(self.cancelled) > 4096:
                self.cancelled.pop(next(iter(self.cancelled)))
            return event is not None
