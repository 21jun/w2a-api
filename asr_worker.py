"""One GPU worker with a hard inference deadline and background recovery."""
from __future__ import annotations

import logging
import math
import multiprocessing as mp
import threading
import time

log = logging.getLogger(__name__)


class ASRServiceError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status = status
        self.code = code


def model_worker(connection, config):
    # CUDA is initialized only in the spawned process, never forked.
    try:
        from server import load_asr
        pipeline = load_asr(config)
        connection.send(("ready", None))
        while True:
            payload = connection.recv()
            started = time.monotonic()
            try:
                result = pipeline.run_asr(**payload)
                connection.send(("ok", (result, time.monotonic() - started)))
            except Exception:
                log.exception("ASR inference failed")
                connection.send(("error", None))
                return  # Reset model state after inference errors, too.
    except EOFError:
        pass
    except Exception:
        log.exception("ASR worker failed")
    finally:
        connection.close()


class ASRWorker:
    def __init__(self, config, timeout_s=10.0, startup_timeout_s=300.0,
                 worker_target=model_worker):
        if not all(math.isfinite(x) and x > 0 for x in (timeout_s, startup_timeout_s)):
            raise ValueError("ASR timeouts must be finite and greater than zero")
        self.config = config
        self.timeout_s = timeout_s
        self.startup_timeout_s = startup_timeout_s
        self.target = worker_target
        self.context = mp.get_context("spawn")
        self.lock = threading.Lock()
        self.process = None
        self.connection = None
        self.closed = False
        self.ready = False

    def _stop(self):
        self.ready = False
        if self.process is not None:
            if self.process.pid is not None:
                self.process.terminate()
                self.process.join(timeout=1)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=1)
                if self.process.is_alive():
                    # Do not start a second model while the old GPU process survives.
                    raise RuntimeError("Could not terminate ASR worker")
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def _start(self):
        if self.closed:
            return
        parent, child = self.context.Pipe()
        self.connection = parent
        self.process = self.context.Process(target=self.target, args=(child, self.config), daemon=True)
        try:
            self.process.start()
            child.close()
            if not parent.poll(self.startup_timeout_s) or parent.recv()[0] != "ready":
                raise RuntimeError("ASR model startup timed out or failed")
            self.ready = True
        except Exception:
            child.close()
            self._stop()
            raise

    def start(self):
        with self.lock:
            self._start()

    def _recover(self):
        with self.lock:
            if self.closed:
                return
            try:
                self._stop()
                self._start()
            except Exception:
                log.exception("ASR model recovery failed; restart the server to retry")

    def run_asr(self, cancel_event=None, **payload):
        from request_control import check_cancelled
        check_cancelled(cancel_event)
        if not self.lock.acquire(blocking=False):
            raise ASRServiceError(503, "ASR_BUSY", "ASR is busy or reloading; try again later.")
        restart = False
        try:
            if self.closed or not self.ready:
                raise ASRServiceError(503, "ASR_UNAVAILABLE", "ASR model is unavailable.")
            check_cancelled(cancel_event)
            self.connection.send(payload)
            deadline = time.monotonic() + self.timeout_s
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    restart = True
                    self._stop()
                    check_cancelled(cancel_event)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    restart = True
                    log.warning("ASR exceeded %.3fs; terminating worker pid=%s", self.timeout_s, self.process.pid)
                    self._stop()
                    raise ASRServiceError(504, "ASR_TIMEOUT", f"ASR exceeded {self.timeout_s:g} seconds; please record again.")
                if self.connection.poll(min(0.05, remaining)):
                    if cancel_event is not None and cancel_event.is_set():
                        restart = True
                        self._stop()
                        check_cancelled(cancel_event)
                    break
            status, value = self.connection.recv()
            if status != "ok":
                raise EOFError("ASR inference failed")
            return value
        except (EOFError, OSError, BrokenPipeError) as exc:
            restart = True
            self._stop()
            raise ASRServiceError(503, "ASR_WORKER_FAILED", "ASR worker failed; model is reloading.") from exc
        finally:
            if restart:
                threading.Thread(target=self._recover, daemon=True).start()
            self.lock.release()

    def close(self):
        with self.lock:
            self.closed = True
            self._stop()
