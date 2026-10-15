import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

import server


def wav(samples, rate=16000, subtype="PCM_16"):
    output = io.BytesIO()
    sf.write(output, np.asarray(samples, dtype=np.float32), rate, format="WAV", subtype=subtype)
    return output.getvalue()


class AudioTests(unittest.TestCase):
    def test_unreadable_empty_and_no_sample_files(self):
        for raw in (b"", b"not audio", wav([])):
            with self.subTest(raw=raw[:20]):
                with self.assertRaises(server.HTTPException) as raised:
                    server.decode_audio(raw, "broken.wav")
                self.assertEqual(raised.exception.status_code, 400)
                self.assertEqual(raised.exception.detail["code"], "INVALID_AUDIO")

    def test_nan_and_infinite_samples_rejected(self):
        for sample in (float("nan"), float("inf")):
            with self.assertRaises(server.HTTPException):
                server.decode_audio(wav([sample], subtype="FLOAT"), "invalid.wav")

    def test_silence_stereo_and_resampling_remain_valid(self):
        for samples, rate in ((np.zeros(1600), 16000), (np.ones((800, 2)) * 0.01, 8000)):
            audio, original_rate = server.decode_audio(wav(samples, rate), "valid.wav")
            self.assertEqual(audio.shape, (1600,))
            self.assertEqual(original_rate, rate)
            self.assertTrue(np.isfinite(audio).all())

    def test_pyav_fallback_is_not_resampled_twice(self):
        with patch.object(server, "decode_with_av", return_value=(np.zeros(1600, dtype=np.float32), 48000)):
            audio, sr = server.decode_audio(b"webm", "recording.webm")
        self.assertEqual((audio.size, sr), (1600, 48000))

    def test_real_pyav_decode_and_resample(self):
        with patch.object(server.sf, "read", side_effect=RuntimeError("try PyAV")):
            audio, sr = server.decode_audio(wav(np.zeros(800), 8000), "fallback.wav")
        self.assertEqual((audio.size, sr), (1600, 8000))

    def test_invalid_target_or_reference_never_calls_asr(self):
        asr = Mock()
        config = SimpleNamespace(top_k=5, retrieval_timeout_s=30)
        with patch.object(server, "ASR", asr), patch.object(server, "CFG", config), patch.object(server, "LEXICON", {"roads": {}}):
            with TestClient(server.app) as client:
                for files in (
                    {"audio": ("bad.wav", b"broken", "audio/wav")},
                    {"audio": ("ok.wav", wav(np.zeros(1600)), "audio/wav"), "ref_audio": ("bad-ref.wav", b"broken", "audio/wav")},
                ):
                    response = client.post("/transcribe", files=files)
                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(response.json()["detail"]["code"], "INVALID_AUDIO")
        asr.run_asr.assert_not_called()

    def test_timeout_returns_504_without_search(self):
        asr = Mock()
        asr.run_asr.side_effect = server.ASRServiceError(504, "ASR_TIMEOUT", "deadline exceeded")
        with patch.object(server, "ASR", asr), patch.object(server, "CFG", SimpleNamespace(top_k=5, retrieval_timeout_s=30)), patch.object(server, "LEXICON", {"roads": {}}), patch.object(server, "retrieve_top_k") as search:
            with TestClient(server.app) as client:
                response = client.post("/transcribe", files={"audio": ("silence.wav", wav(np.zeros(1600)), "audio/wav")})
            self.assertEqual(response.status_code, 504)
            self.assertEqual(response.json()["detail"]["code"], "ASR_TIMEOUT")
            asr.run_asr.assert_called_once()
            search.assert_not_called()

    def test_valid_request_preserves_response_and_worker_timing(self):
        asr = Mock()
        asr.run_asr.return_value = ("recognized", 0.25)
        with patch.object(server, "ASR", asr), patch.object(server, "CFG", SimpleNamespace(top_k=5, retrieval_timeout_s=30)), patch.object(server, "LEXICON", {"roads": {}}), patch.object(server, "retrieve_top_k", return_value=([], "ipa")):
            with TestClient(server.app) as client:
                response = client.post("/transcribe", files={"audio": ("silence.wav", wav(np.zeros(1600)), "audio/wav")})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["asr_result"], "recognized")
            self.assertEqual(response.json()["timing"]["asr_s"], 0.25)


if __name__ == "__main__":
    unittest.main()
