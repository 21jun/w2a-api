import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from asr_worker import ASRServiceError
from request_control import RequestRegistry
import server
from test_audio_validation import wav


class ControlTests(unittest.TestCase):
    def test_cancel_before_request_and_between_stages(self):
        registry = RequestRegistry()
        self.assertFalse(registry.cancel("early"))
        with self.assertRaises(ASRServiceError):
            with registry.track("early"):
                self.fail("cancelled request was started")
        with registry.track("stages"):
            self.assertTrue(registry.cancel("stages"))
        with self.assertRaises(ASRServiceError):
            with registry.track("stages"):
                self.fail("second stage was started after cancellation")

    def test_wrong_id_and_duplicate_id(self):
        registry = RequestRegistry()
        with registry.track("one") as token:
            self.assertFalse(registry.cancel("other"))
            self.assertFalse(token.is_set())
            with self.assertRaises(ASRServiceError) as raised:
                with registry.track("one"):
                    pass
            self.assertEqual(raised.exception.status, 409)
        self.assertFalse(registry.active)

    def test_retrieval_timeout_stops_iteration(self):
        entries = {str(i): {"ipa": "a", "entity_id": str(i), "entity": str(i), "romanized": ""} for i in range(100)}
        def distance(*args):
            time.sleep(0.02)
            return 0
        with patch.object(server.epi, "transliterate", return_value="a"), patch.object(server.dst, "levenshtein_distance", side_effect=distance) as measured:
            with self.assertRaises(ASRServiceError) as raised:
                server.retrieve_top_k("text", entries, deadline=time.monotonic() + 0.05)
            self.assertEqual(raised.exception.code, "RETRIEVAL_TIMEOUT")
            self.assertLess(measured.call_count, len(entries))

    def test_retrieval_cancel_stops_iteration(self):
        token = threading.Event()
        entry = {"ipa": "a", "entity_id": "1", "entity": "word", "romanized": ""}
        def distance(*args):
            token.set()
            return 0
        with patch.object(server.epi, "transliterate", return_value="a"), patch.object(server.dst, "levenshtein_distance", side_effect=distance):
            with self.assertRaises(ASRServiceError) as raised:
                server.retrieve_top_k("text", {"word": entry}, cancel_event=token)
            self.assertEqual(raised.exception.code, "REQUEST_CANCELLED")

    def test_cancel_endpoint_signals_active_asr(self):
        entered = threading.Event()
        def run(**payload):
            entered.set()
            self.assertTrue(payload["cancel_event"].wait(3))
            raise ASRServiceError(499, "REQUEST_CANCELLED", "cancelled")
        asr = Mock()
        asr.run_asr.side_effect = run
        with patch.object(server, "ASR", asr), patch.object(server, "CFG", SimpleNamespace(top_k=5, retrieval_timeout_s=30)), patch.object(server, "LEXICON", {"roads": {}}), patch.object(server, "REQUESTS", RequestRegistry()):
            with TestClient(server.app) as client, ThreadPoolExecutor() as pool:
                pending = pool.submit(client.post, "/transcribe", files={"audio": ("audio.wav", wav([0] * 160), "audio/wav")}, data={"request_id": "r1"})
                self.assertTrue(entered.wait(3))
                response = client.post("/cancel", data={"request_id": "r1"})
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json()["was_running"])
                result = pending.result(timeout=3)
                self.assertEqual(result.status_code, 499)
                self.assertEqual(result.json()["detail"]["code"], "REQUEST_CANCELLED")

    def test_retrieve_endpoint_timeout(self):
        def transliterate(text):
            time.sleep(0.02)
            return "a"
        with patch.object(server, "CFG", SimpleNamespace(top_k=5, retrieval_timeout_s=0.001)), patch.object(server, "LEXICON", {"roads": {}}), patch.object(server.epi, "transliterate", side_effect=transliterate):
            with TestClient(server.app) as client:
                response = client.post("/retrieve", data={"text": "hello"})
        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json()["detail"]["code"], "RETRIEVAL_TIMEOUT")

    def test_cancel_endpoint_stops_running_retrieval(self):
        entered = threading.Event()
        def distance(*args):
            entered.set()
            time.sleep(0.05)
            return 0
        entry = {"ipa": "a", "entity_id": "1", "entity": "word", "romanized": ""}
        entries = {str(i): entry for i in range(100)}
        with patch.object(server, "CFG", SimpleNamespace(top_k=5, retrieval_timeout_s=30)), patch.object(server, "LEXICON", {"roads": entries}), patch.object(server, "REQUESTS", RequestRegistry()), patch.object(server.epi, "transliterate", return_value="a"), patch.object(server.dst, "levenshtein_distance", side_effect=distance) as measured:
            with TestClient(server.app) as client, ThreadPoolExecutor() as pool:
                pending = pool.submit(client.post, "/retrieve", data={"text": "hello", "request_id": "search"})
                self.assertTrue(entered.wait(3))
                response = client.post("/cancel", data={"request_id": "search"})
                self.assertTrue(response.json()["was_running"])
                self.assertEqual(pending.result(timeout=3).status_code, 499)
                self.assertLess(measured.call_count, len(entries))

    def test_asr_only_skips_search(self):
        asr = Mock()
        asr.run_asr.return_value = ("recognized", 0.1)
        with patch.object(server, "ASR", asr), patch.object(server, "CFG", SimpleNamespace(top_k=5)), patch.object(server, "LEXICON", {"roads": {}}), patch.object(server, "retrieve_top_k") as search:
            with TestClient(server.app) as client:
                response = client.post("/transcribe", files={"audio": ("audio.wav", wav([0] * 160), "audio/wav")}, data={"with_retrieval": "false"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["retrieved"], [])
        search.assert_not_called()

    def test_reference_text_is_preserved(self):
        # Exercise the real method without loading torch or the GPU checkpoint.
        source = Path("wake2adapt/src/infer_qwen2.5_omni.py")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen25OmniASRPipeline")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "run_asr")
        namespace = {"TRANSCRIPTION_PREFIX": "Transcription: "}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        for output in ("상곡안길", "상곡안길입니다"):
            fake = SimpleNamespace(_normalize_audio_input=lambda audio: audio, _build_messages=lambda *args, **kwargs: [], _transcribe_from_messages=lambda *args: "Transcription: " + output, config=SimpleNamespace(max_new_tokens=256))
            self.assertEqual(namespace["run_asr"](fake, [], reference_text="상곡안길"), output)

    def test_empty_transcript_never_starts_phonetic_search(self):
        with patch.object(server.epi, "transliterate") as transliterate:
            for text in ("", "  \n"):
                with self.assertRaises(ASRServiceError) as raised:
                    server.retrieve_top_k(text, {})
                self.assertEqual(raised.exception.code, "EMPTY_TRANSCRIPT")
            transliterate.assert_not_called()

    def test_empty_asr_result_returns_search_failure_before_search(self):
        asr = Mock()
        asr.run_asr.return_value = ("", 0.1)
        with patch.object(server, "ASR", asr), patch.object(server, "CFG", SimpleNamespace(top_k=5, retrieval_timeout_s=10)), patch.object(server, "LEXICON", {"roads": {}}), patch.object(server.epi, "transliterate") as search:
            with TestClient(server.app) as client:
                response = client.post("/transcribe", files={"audio": ("audio.wav", wav([0] * 160), "audio/wav")})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["detail"]["code"], "EMPTY_TRANSCRIPT")
            search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
