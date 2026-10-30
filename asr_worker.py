"""One GPU worker with a hard inference deadline and background recovery."""
from __future__ import annotations

import logging
import math
import multiprocessing as mp
import sys
import threading
import time
import traceback

log = logging.getLogger(__name__)


class ASRServiceError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status = status
        self.code = code


def model_worker(connection, config):
    # CUDA is initialized only in the spawned process, never forked.
    ready = False
    try:
        from server import load_asr
        pipeline = load_asr(config)
        connection.send(("ready", None))
        ready = True
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
    except BaseException as exc:
        if not ready:
            # Send the original startup exception before closing the pipe. Parent stderr
            # may be redirected or the child may exit before its logs are flushed.
            detail = traceback.format_exc()
            try:
                connection.send(("startup_error", detail))
            except (EOFError, OSError):
                pass
            log.error("ASR model startup failed:\n%s", detail)
        elif not isinstance(exc, EOFError):
            log.exception("ASR worker failed")
    finally:
        connection.close()


class ASRWorker:
    def __init__(self, config, timeout_s=10.0, worker_target=model_worker):
        if not (math.isfinite(timeout_s) and timeout_s > 0):
            raise ValueError("ASR timeout must be finite and greater than zero")
        self.config = config
        self.timeout_s = timeout_s
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
            try:
                # No startup deadline: the first run may download the model weights.
                status, detail = parent.recv()
            except (EOFError, OSError) as exc:
                # A native crash or spawn/bootstrap failure cannot send an error message.
                self.process.join(timeout=0.5)
                exitcode = self.process.exitcode
                code = "not available yet" if exitcode is None else f"{exitcode} (0x{exitcode & 0xffffffff:08X})"
                raise RuntimeError(
                    f"ASR worker exited before reporting ready. Exit code: {code}. "
                    f"Python: {sys.executable}. Check earlier child-process stderr for "
                    "spawn/import errors or native DLL/CUDA failures."
                ) from exc
            if status == "startup_error":
                raise RuntimeError(f"ASR model startup failed in {sys.executable}:\n{detail}")
            if status != "ready":
                raise RuntimeError(f"Unexpected ASR startup response: {status!r}")
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
