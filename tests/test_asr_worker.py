import time
import threading
import unittest

from asr_worker import ASRServiceError, ASRWorker


def fake_worker(connection, config):
    connection.send(("ready", None))
    try:
        while True:
            payload = connection.recv()
            if payload.get("crash"):
                return
            time.sleep(payload.get("delay", 0))
            connection.send(("ok", ("recognized", 0.01)))
    except EOFError:
        pass
    finally:
        connection.close()


class WorkerTests(unittest.TestCase):
    def worker(self, timeout=1):
        worker = ASRWorker(None, timeout, 5, worker_target=fake_worker)
        self.addCleanup(worker.close)
        worker.start()
        return worker

    def wait_ready(self, worker):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if worker.ready:
                return
            time.sleep(0.02)
        self.fail("worker did not recover")

    def test_success(self):
        self.assertEqual(self.worker().run_asr(), ("recognized", 0.01))

    def test_timeout_kills_worker_and_recovers(self):
        worker = self.worker(0.15)
        old_pid = worker.process.pid
        started = time.monotonic()
        with self.assertRaises(ASRServiceError) as raised:
            worker.run_asr(delay=60)
        self.assertEqual((raised.exception.status, raised.exception.code), (504, "ASR_TIMEOUT"))
        self.assertLess(time.monotonic() - started, 3)
        self.wait_ready(worker)
        self.assertNotEqual(worker.process.pid, old_pid)
        self.assertEqual(worker.run_asr()[0], "recognized")

    def test_crashed_worker_recovers(self):
        worker = self.worker()
        with self.assertRaises(ASRServiceError) as raised:
            worker.run_asr(crash=True)
        self.assertEqual(raised.exception.code, "ASR_WORKER_FAILED")
        self.wait_ready(worker)
        self.assertEqual(worker.run_asr()[0], "recognized")

    def test_busy_does_not_queue_indefinitely(self):
        worker = self.worker()
        worker.lock.acquire()
        try:
            with self.assertRaises(ASRServiceError) as raised:
                worker.run_asr()
            self.assertEqual((raised.exception.status, raised.exception.code), (503, "ASR_BUSY"))
        finally:
            worker.lock.release()

    def test_invalid_timeouts(self):
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                ASRWorker(None, timeout)

    def test_cancel_running_inference_kills_and_recovers(self):
        worker = self.worker(5)
        old_pid = worker.process.pid
        token = threading.Event()
        errors = []

        def run():
            try:
                worker.run_asr(cancel_event=token, delay=60)
            except ASRServiceError as exc:
                errors.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        time.sleep(0.1)
        token.set()
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors[0].code, "REQUEST_CANCELLED")
        self.wait_ready(worker)
        self.assertNotEqual(worker.process.pid, old_pid)
        self.assertEqual(worker.run_asr()[0], "recognized")

    def test_cancel_before_inference_keeps_model(self):
        worker = self.worker()
        pid = worker.process.pid
        token = threading.Event()
        token.set()
        with self.assertRaises(ASRServiceError) as raised:
            worker.run_asr(cancel_event=token)
        self.assertEqual(raised.exception.status, 499)
        self.assertEqual(worker.process.pid, pid)


if __name__ == "__main__":
    unittest.main()
