"""
wake2adapt serving API: POST an audio file, get back the Qwen2.5-Omni ASR result and the
phonetic-edit-distance retrieval hits over the L2-KPNS entity lexicon.

# start (loads Qwen2.5-Omni on GPU)
python server.py --port 8000

# retrieval only, no GPU / no model load (handy for checking the retriever)
python server.py --port 8000 --no-asr

# zero-shot
curl -s -X POST localhost:8000/transcribe -F audio=@sample.wav -F domain=roads | jq

# 1-shot ASR adaptation (reference audio from the same speaker)
curl -s -X POST localhost:8000/transcribe \
     -F audio=@sample.wav -F domain=roads \
     -F ref_audio=@RD16245_P001.wav -F ref_text=상곡안길 | jq

# retriever only, text in
curl -s -X POST localhost:8000/retrieve -F text=상곡안길 -F domain=roads | jq

Browser UI with the same fields: http://localhost:8000/
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import csv
import heapq
import importlib.util
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict

import av
import epitran
import numpy as np
import panphon.distance
import soundfile as sf
import soxr
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from asr_worker import ASRServiceError, ASRWorker
from request_control import RequestRegistry, check_cancelled

TARGET_SR = 16000
DOMAINS = ["roads", "content", "restaurants", "stations"]

epi = epitran.Epitran("kor-Hang")
dst = panphon.distance.Distance()


class LexiconEntry(TypedDict):
    entity_id: str
    entity: str
    romanized: str
    ipa: str


# filled in by main()
CFG: "ServerConfig" = None
LEXICON: Dict[str, Dict[str, LexiconEntry]] = {}  # domain -> {Korean entity: metadata}
ASR = None                                       # ASRWorker, None with --no-asr
REQUESTS = RequestRegistry()


class ServerConfig:
    def __init__(self, args: argparse.Namespace):
        self.repo_root = Path(args.repo_root).resolve()
        self.jsonl_dir = Path(args.jsonl_dir or self.repo_root / "L2-KPNS-jsonl").resolve()
        self.lexicon_csv_dir = Path(args.lexicon_csv_dir).resolve() if args.lexicon_csv_dir else None
        self.domains = [d.strip() for d in args.domains.split(",") if d.strip()]
        self.model_path = args.model_path
        self.device = args.device
        self.dtype = args.dtype
        self.max_new_tokens = args.max_new_tokens
        self.top_k = args.top_k
        self.no_asr = args.no_asr
        self.quant = args.quant
        self.keep_visual = args.keep_visual
        self.keep_audio_output = args.keep_audio_output
        self.asr_timeout_s = args.asr_timeout_s
        self.asr_startup_timeout_s = args.asr_startup_timeout_s
        self.retrieval_timeout_s = args.retrieval_timeout_s


# --------------------------------------------------------------------------------------
# retrieval (same scoring as wake2adapt/src/infer_qwen2.5_omni.py)
# --------------------------------------------------------------------------------------

def normalized_phoneme_editdistance(dist: float, utterance_ipa: str, entity_ipa: str) -> float:
    return 1 - (dist / max(len(utterance_ipa), len(entity_ipa), 1))


def retrieve_top_k(
    utterance: str,
    entities: Dict[str, LexiconEntry],
    k: int = 10,
    *,
    deadline: Optional[float] = None,
    cancel_event=None,
) -> Tuple[List[dict], str]:
    """Top-k entities by phoneme-level edit distance against the ASR result."""
    def checkpoint():
        check_cancelled(cancel_event)
        if deadline is not None and time.monotonic() >= deadline:
            raise ASRServiceError(504, "RETRIEVAL_TIMEOUT", "DB 검색 제한 시간을 초과했습니다.")

    checkpoint()
    if not utterance.strip():
        raise ASRServiceError(400, "EMPTY_TRANSCRIPT", "DB 검색 실패: ASR 결과가 비어 있어 검색할 수 없습니다.")
    utterance_ipa = epi.transliterate(utterance)
    checkpoint()
    scored = []
    for position, entry in enumerate(entities.values()):
        checkpoint()
        ipa = entry["ipa"]
        dist = int(dst.levenshtein_distance(utterance_ipa, ipa))  # panphon hands back np.int64
        checkpoint()
        row = {
            "entity_id": entry["entity_id"],
            "entity": entry["entity"],
            "romanized": entry["romanized"],
            "score": float(round(normalized_phoneme_editdistance(dist, utterance_ipa, ipa), 2)),
            "distance": dist,
            "ipa": ipa,
        }
        if len(scored) < k:
            heapq.heappush(scored, (row["score"], -position, row))
        elif (row["score"], -position) > scored[0][:2]:
            heapq.heapreplace(scored, (row["score"], -position, row))
    top = [row for _, _, row in sorted(scored, reverse=True)]
    for rank, row in enumerate(top, start=1):
        row["rank"] = rank
    checkpoint()
    return top, utterance_ipa


def load_lexicon(cfg: ServerConfig) -> Dict[str, Dict[str, LexiconEntry]]:
    """Load full L2-KPNS CSV metadata, with the legacy 200-item data as a fallback."""
    lexicon: Dict[str, Dict[str, LexiconEntry]] = {}
    for domain in cfg.domains:
        csv_path = None
        if cfg.lexicon_csv_dir:
            for filename in (f"{domain}.csv", f"{domain}_200.csv"):
                candidate = cfg.lexicon_csv_dir / filename
                if candidate.exists():
                    csv_path = candidate
                    break

        table: Dict[str, LexiconEntry] = {}
        if csv_path and csv_path.exists():
            with open(csv_path, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or "korean" not in reader.fieldnames:
                    raise ValueError(f"lexicon CSV needs a korean column: {csv_path}")
                for row in reader:
                    entity = (row.get("korean") or "").strip()
                    if not entity or entity in table:
                        continue
                    table[entity] = {
                        "entity_id": (row.get("entity_id") or "").strip(),
                        "entity": entity,
                        "romanized": (row.get("romanized") or "").strip(),
                        "ipa": (row.get("phonemes") or "").strip()
                        or epi.transliterate(entity),
                    }
            source = str(csv_path)
        else:
            names: List[str] = []
            for path in sorted(cfg.jsonl_dir.glob(f"*/{domain}_*.jsonl")):
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            names.append(json.loads(line)["answer"])
            source = f"{cfg.jsonl_dir}/*/{domain}_*.jsonl"
            for entity in dict.fromkeys(names):
                table[entity] = {
                    "entity_id": "",
                    "entity": entity,
                    "romanized": "",
                    "ipa": epi.transliterate(entity),
                }

        if not table:
            print(f"[warn] no entities found for domain={domain} ({source})")
            continue
        lexicon[domain] = table
        print(f"[lexicon] {domain}: {len(table)} entities from {source}")
    return lexicon


def entity_ipa_for(domain: str) -> Dict[str, LexiconEntry]:
    """A single domain, or every domain merged when domain == 'all'."""
    if domain == "all":
        merged: Dict[str, LexiconEntry] = {}
        for table in LEXICON.values():
            merged.update(table)
        return merged
    if domain not in LEXICON:
        raise HTTPException(400, f"unknown domain '{domain}' (have: {list(LEXICON)} or 'all')")
    return LEXICON[domain]


# --------------------------------------------------------------------------------------
# ASR
# --------------------------------------------------------------------------------------

class ModelLoader:
    """Stands in for the model class inside the repo module, so extra load options reach
    from_pretrained without editing wake2adapt's code (it only ever calls from_pretrained)."""

    def __init__(self, cls, extra: dict):
        self.cls = cls
        self.extra = extra

    def from_pretrained(self, *args, **kwargs):
        return self.cls.from_pretrained(*args, **{**kwargs, **self.extra})


def quantization_config(cfg: ServerConfig, compute_dtype):
    """bitsandbytes config for --quant. The audio encoder and the output head stay in full
    precision: together they are ~2 GiB and quantizing them costs entity accuracy."""
    from transformers import BitsAndBytesConfig

    skip = ["thinker.audio_tower", "thinker.lm_head", "talker", "token2wav"]
    if cfg.quant == "int8":
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
        llm_int8_skip_modules=skip,
    )


def load_asr(cfg: ServerConfig):
    """Load wake2adapt's pipeline by path (the filename has a '.' in it, so no plain import)."""
    import torch

    if str(cfg.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested {cfg.device}, but CUDA is unavailable in {sys.executable} "
            f"(torch={torch.__version__}, built CUDA={torch.version.cuda}). "
            "Use a Python environment with CUDA-enabled PyTorch and a compatible NVIDIA GPU/driver. "
            "Use --no-asr only if you need retrieval without ASR."
        )

    infer_py = cfg.repo_root / "src" / "infer_qwen2.5_omni.py"
    if not infer_py.exists():
        raise FileNotFoundError(f"{infer_py} not found -- pass --repo_root")
    spec = importlib.util.spec_from_file_location("infer_qwen2_5_omni", infer_py)
    module = importlib.util.module_from_spec(spec)
    sys.modules["infer_qwen2_5_omni"] = module  # dataclass forward refs look here
    sys.path.insert(0, str(cfg.repo_root))
    spec.loader.exec_module(module)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.dtype]
    asr_cfg = module.ASRConfig(
        model_path=cfg.model_path,
        device=cfg.device,
        torch_dtype=dtype,
        max_new_tokens=cfg.max_new_tokens,
    )
    extra = {}
    if not cfg.keep_audio_output:
        # the talker/token2wav stack (~4.2 GiB) only generates speech; ASR never touches it
        extra["enable_audio_output"] = False
    if cfg.quant != "none":
        extra["quantization_config"] = quantization_config(cfg, dtype)
    module.Qwen2_5OmniForConditionalGeneration = ModelLoader(
        module.Qwen2_5OmniForConditionalGeneration, extra
    )

    print(f"[asr] loading {cfg.model_path} on {cfg.device} "
          f"({cfg.dtype}, quant={cfg.quant}) ...")
    pipeline = module.Qwen25OmniASRPipeline(asr_cfg)
    if not cfg.keep_visual:
        pipeline.model.thinker.visual = None  # 1.26 GiB vision tower, unused for audio-only ASR
        torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated(cfg.device) / 2**30 if torch.cuda.is_available() else 0
    print(f"[asr] ready ({allocated:.2f} GiB allocated)")
    return pipeline


def decode_with_av(raw: bytes) -> Tuple[np.ndarray, int]:
    """Decode with PyAV (bundled ffmpeg libs) straight to mono float32 @ 16 kHz.

    Browser recordings arrive as webm/opus or mp4/aac, which libsndfile cannot read.
    librosa's fallback is no help here: it wants an ffmpeg binary (absent) and numba
    cannot cache its kernels from this filesystem."""
    with av.open(io.BytesIO(raw)) as container:
        stream = container.streams.audio[0]
        orig_sr = stream.rate or TARGET_SR
        resampler = av.audio.resampler.AudioResampler(
            format="flt", layout="mono", rate=TARGET_SR
        )
        chunks = [
            resampled.to_ndarray().reshape(-1)
            for frame in container.decode(audio=0)
            for resampled in resampler.resample(frame)
        ]
        chunks += [r.to_ndarray().reshape(-1) for r in resampler.resample(None)]  # flush
    if not chunks:
        raise ValueError("no decodable audio frames")
    return np.concatenate(chunks).astype(np.float32), orig_sr


def decode_audio(raw: bytes, filename: str) -> Tuple[np.ndarray, int]:
    """bytes -> (mono float32 @ 16 kHz, original sample rate)."""
    try:
        if not raw:
            raise ValueError("empty file")
        try:
            audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        except Exception:
            audio, sr = decode_with_av(raw)
            needs_resampling = False  # PyAV already returns 16 kHz.
        else:
            needs_resampling = sr != TARGET_SR
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float32)
        if sr <= 0 or audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
            raise ValueError("missing or invalid audio samples")
        if needs_resampling:
            audio = soxr.resample(audio, sr, TARGET_SR).astype(np.float32)
        if not audio.size or not np.isfinite(audio).all():
            raise ValueError("invalid resampled audio")
        return audio, sr
    except Exception as exc:
        raise HTTPException(400, {
            "code": "INVALID_AUDIO",
            "message": "음성 파일을 읽을 수 없습니다. 파일 손상 또는 지원되지 않는 형식일 수 있습니다.",
            "filename": filename,
        }) from exc


# --------------------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------------------

app = FastAPI(title="wake2adapt serving API")


@app.exception_handler(ASRServiceError)
async def service_error_handler(request, exc):
    return JSONResponse(status_code=exc.status, content={"detail": {"code": exc.code, "message": str(exc)}})


@app.post("/cancel")
def cancel(request_id: str = Form(..., pattern=r"^[\w-]{1,64}$")):
    return {"request_id": request_id, "was_running": REQUESTS.cancel(request_id)}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "asr_loaded": ASR is not None and ASR.ready,
        "asr_timeout_s": CFG.asr_timeout_s,
        "retrieval_timeout_s": CFG.retrieval_timeout_s,
        "model_path": None if ASR is None else CFG.model_path,
        "device": None if ASR is None else CFG.device,
        "dtype": None if ASR is None else CFG.dtype,
        "quant": None if ASR is None else CFG.quant,
        "domains": {d: len(t) for d, t in LEXICON.items()},
        "default_top_k": CFG.top_k,
    }


@app.get("/lexicon")
def lexicon(domain: str = "roads", limit: int = 20):
    table = entity_ipa_for(domain)
    items = list(table.values())[:limit]
    return {"domain": domain, "size": len(table), "items": items}


@app.post("/retrieve")
def retrieve(text: str = Form(...), domain: str = Form("roads"), top_k: Optional[int] = Form(None),
             request_id: Optional[str] = Form(None, pattern=r"^[\w-]{1,64}$")):
    """Retriever only -- feed it a transcription directly, no GPU involved."""
    k = top_k or CFG.top_k
    t0 = time.perf_counter()
    with REQUESTS.track(request_id) as token:
        hits, text_ipa = retrieve_top_k(text, entity_ipa_for(domain), k=k,
                                      deadline=time.monotonic() + CFG.retrieval_timeout_s,
                                      cancel_event=token)
    return {
        "text": text,
        "text_ipa": text_ipa,
        "domain": domain,
        "top_k": k,
        "retrieved": hits,
        "retr_entities": [h["entity"] for h in hits],
        "timing": {"retrieval_s": round(time.perf_counter() - t0, 3)},
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(..., description="audio to transcribe (wav/flac/mp3/...)"),
    domain: str = Form("roads"),
    top_k: Optional[int] = Form(None),
    ref_audio: Optional[UploadFile] = File(None, description="1-shot reference audio (same speaker)"),
    ref_text: str = Form(""),
    request_id: Optional[str] = Form(None, pattern=r"^[\w-]{1,64}$"),
    with_retrieval: bool = Form(True),
):
    """ASR (zero-shot, or 1-shot adaptation when ref_audio is sent) + phonetic retrieval."""
    if ASR is None:
        raise HTTPException(503, "server started with --no-asr; only /retrieve is available")
    with REQUESTS.track(request_id) as token:
        return await _transcribe(audio, domain, top_k, ref_audio, ref_text, with_retrieval, token)


async def _transcribe(audio, domain, top_k, ref_audio, ref_text, with_retrieval, token):
    k = top_k or CFG.top_k
    entities = entity_ipa_for(domain)

    t0 = time.perf_counter()
    check_cancelled(token)
    wav, orig_sr = await asyncio.to_thread(decode_audio, await audio.read(), audio.filename or "audio")
    check_cancelled(token)
    ref_wav = None
    if ref_audio is not None and ref_audio.filename:
        ref_wav, _ = await asyncio.to_thread(decode_audio, await ref_audio.read(), ref_audio.filename)
    t_decode = time.perf_counter()
    check_cancelled(token)

    try:
        asr_result, asr_seconds = await asyncio.to_thread(
            ASR.run_asr, audio=wav, ref_audio=ref_wav, reference_text=ref_text, cancel_event=token,
        )
    except ASRServiceError as exc:
        raise HTTPException(exc.status, {"code": exc.code, "message": str(exc)}) from exc
    t_asr = time.perf_counter()
    check_cancelled(token)

    if with_retrieval:
        hits, asr_ipa = await asyncio.to_thread(
            retrieve_top_k, asr_result, entities, k=k,
            deadline=time.monotonic() + CFG.retrieval_timeout_s, cancel_event=token,
        )
    else:
        hits, asr_ipa = [], ""
    check_cancelled(token)
    t_end = time.perf_counter()

    return {
        "asr_result": asr_result,
        "asr_ipa": asr_ipa,
        "asr_adaptation": ref_wav is not None,
        "ref_text": ref_text,
        "domain": domain,
        "top_k": k,
        "lexicon_size": len(entities),
        "retrieved": hits,
        "retr_entities": [h["entity"] for h in hits],
        "audio": {
            "filename": audio.filename,
            "orig_sample_rate": orig_sr,
            "seconds": round(len(wav) / TARGET_SR, 2),
        },
        "timing": {
            "decode_s": round(t_decode - t0, 3),
            "asr_s": round(asr_seconds, 3),
            "retrieval_s": round(t_end - t_asr, 3),
            "total_s": round(t_end - t0, 3),
        },
    }


INDEX_HTML = """<!doctype html>
<meta charset="utf-8"><title>wake2adapt</title>
<style>
 body{font:14px/1.5 system-ui,sans-serif;max-width:860px;margin:32px auto;padding:0 16px}
 label{display:block;margin:10px 0 2px;font-weight:600}
 input,select,button{font:inherit;padding:6px}
 button{margin-top:16px;cursor:pointer}
 table{border-collapse:collapse;width:100%;margin-top:12px}
 th,td{border:1px solid #ddd;padding:4px 8px;text-align:left}
 th{background:#f4f4f4} pre{background:#f7f7f7;padding:12px;overflow:auto}
 .big{font-size:18px;font-weight:600;margin-top:16px}
</style>
<h2>wake2adapt - ASR + phonetic retrieval</h2>
<form id="f">
 <label>audio <input type="file" name="audio" accept="audio/*" required></label>
 <label>domain <select name="domain">
   <option>roads</option><option>content</option><option>restaurants</option>
   <option>stations</option><option value="all">all</option></select></label>
 <label>top_k <input type="number" name="top_k" value="10" min="1" max="50"></label>
 <label>ref_audio (1-shot, optional) <input type="file" name="ref_audio" accept="audio/*"></label>
 <label>ref_text <input type="text" name="ref_text" placeholder="상곡안길"></label>
 <button type="submit">transcribe</button>
</form>
<div id="out"></div>
<script>
document.getElementById('f').onsubmit = async (e) => {
  e.preventDefault();
  const out = document.getElementById('out');
  out.innerHTML = 'running...';
  const fd = new FormData(e.target);
  const ref = fd.get('ref_audio');
  if (!ref || !ref.size) fd.delete('ref_audio');
  const r = await fetch('/transcribe', {method: 'POST', body: fd});
  const j = await r.json();
  if (!r.ok) { out.innerHTML = '<pre>' + JSON.stringify(j, null, 2) + '</pre>'; return; }
  const rows = j.retrieved.map(h =>
    `<tr><td>${h.rank}</td><td>${h.entity}</td><td>${h.score}</td><td>${h.distance}</td><td>${h.ipa}</td></tr>`).join('');
  out.innerHTML = `<div class="big">ASR: ${j.asr_result}</div>
    <div>ipa: ${j.asr_ipa} · adaptation: ${j.asr_adaptation} · ${j.audio.seconds}s ·
         asr ${j.timing.asr_s}s / retrieval ${j.timing.retrieval_s}s</div>
    <table><tr><th>#</th><th>entity</th><th>score</th><th>dist</th><th>ipa</th></tr>${rows}</table>
    <details><summary>raw json</summary><pre>${JSON.stringify(j, null, 2)}</pre></details>`;
};
</script>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


def main():
    global CFG, LEXICON, ASR
    server_root = Path(__file__).resolve().parent
    default_repo = server_root / "wake2adapt"
    default_lexicon_dir = next(
        (
            path
            for path in (
                server_root / "L2-KPNS" / "metadata" / "lexicons",
                server_root.parent / "L2-KPNS" / "metadata" / "lexicons",
            )
            if path.is_dir()
        ),
        None,
    )

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--repo_root", default=os.environ.get("W2A_REPO", str(default_repo)),
                   help="wake2adapt checkout (default: ./wake2adapt next to this file)")
    p.add_argument("--jsonl_dir", default=None, help="default: {repo_root}/L2-KPNS-jsonl")
    p.add_argument("--lexicon_csv_dir",
                   default=os.environ.get(
                       "W2A_LEXICON_DIR",
                       str(default_lexicon_dir) if default_lexicon_dir else None,
                   ),
                   help="directory holding the full L2-KPNS {domain}.csv files; "
                        "falls back to {domain}_200.csv, then the JSONL answers")
    p.add_argument("--domains", default=",".join(DOMAINS))
    p.add_argument("--model_path", default=os.environ.get("W2A_MODEL", "Qwen/Qwen2.5-Omni-7B"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--asr_timeout_s", type=float, default=10.0,
                   help="hard deadline per ASR call, excluding model load and retrieval (default: 10 s)")
    p.add_argument("--asr_startup_timeout_s", type=float, default=300.0,
                   help="model startup/reload deadline (default: 300 s)")
    p.add_argument("--retrieval_timeout_s", type=float, default=20.0,
                   help="phonetic DB search deadline (default: 20 s)")
    p.add_argument("--quant", default="none", choices=["none", "int8", "nf4"],
                   help="bitsandbytes weight quantization; nf4 fits an 11 GB card "
                        "(needs compute capability >= 7.5)")
    p.add_argument("--keep_visual", action="store_true",
                   help="keep the unused vision tower on the GPU (+1.26 GiB)")
    p.add_argument("--keep_audio_output", action="store_true",
                   help="load the talker/token2wav speech stack too (+4.2 GiB at load time)")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--no-asr", "--no_asr", dest="no_asr", action="store_true",
                   help="skip loading the model; only /retrieve and /lexicon work")
    args = p.parse_args()
    if not math.isfinite(args.retrieval_timeout_s) or args.retrieval_timeout_s <= 0:
        p.error("--retrieval_timeout_s must be finite and greater than zero")

    CFG = ServerConfig(args)
    LEXICON = load_lexicon(CFG)
    if not CFG.no_asr:
        ASR = ASRWorker(CFG, CFG.asr_timeout_s, CFG.asr_startup_timeout_s)
        atexit.register(ASR.close)
        ASR.start()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
