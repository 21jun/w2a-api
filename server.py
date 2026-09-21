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
import csv
import importlib.util
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import av
import epitran
import numpy as np
import panphon.distance
import soundfile as sf
import soxr
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

TARGET_SR = 16000
DOMAINS = ["roads", "content", "restaurants", "stations"]

epi = epitran.Epitran("kor-Hang")
dst = panphon.distance.Distance()

# filled in by main()
CFG: "ServerConfig" = None
LEXICON: Dict[str, Dict[str, str]] = {}   # domain -> {entity: ipa}
ASR = None                                # Qwen25OmniASRPipeline, None with --no-asr


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


# --------------------------------------------------------------------------------------
# retrieval (same scoring as wake2adapt/src/infer_qwen2.5_omni.py)
# --------------------------------------------------------------------------------------

def normalized_phoneme_editdistance(dist: float, utterance_ipa: str, entity_ipa: str) -> float:
    return 1 - (dist / max(len(utterance_ipa), len(entity_ipa), 1))


def retrieve_top_k(utterance: str, entity_ipa: Dict[str, str], k: int = 10) -> Tuple[List[dict], str]:
    """Top-k entities by phoneme-level edit distance against the ASR result."""
    utterance_ipa = epi.transliterate(utterance)
    scored = []
    for entity, ipa in entity_ipa.items():
        dist = int(dst.levenshtein_distance(utterance_ipa, ipa))  # panphon hands back np.int64
        scored.append({
            "entity": entity,
            "score": float(round(normalized_phoneme_editdistance(dist, utterance_ipa, ipa), 2)),
            "distance": dist,
            "ipa": ipa,
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:k]
    for rank, row in enumerate(top, start=1):
        row["rank"] = rank
    return top, utterance_ipa


def load_lexicon(cfg: ServerConfig) -> Dict[str, Dict[str, str]]:
    """domain -> {entity: ipa}. Prefers the L2-KPNS {domain}_200.csv lexicon when its directory
    is given, otherwise rebuilds the 200 entities per domain from the answers in
    L2-KPNS-jsonl (same set: 200 unique answers per domain)."""
    lexicon: Dict[str, Dict[str, str]] = {}
    for domain in cfg.domains:
        names: List[str] = []
        csv_path = cfg.lexicon_csv_dir / f"{domain}_200.csv" if cfg.lexicon_csv_dir else None
        if csv_path and csv_path.exists():
            with open(csv_path, encoding="utf-8-sig") as f:
                names = [row["korean"] for row in csv.DictReader(f)]
            source = str(csv_path)
        else:
            for path in sorted(cfg.jsonl_dir.glob(f"*/{domain}_*.jsonl")):
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            names.append(json.loads(line)["answer"])
            source = f"{cfg.jsonl_dir}/*/{domain}_*.jsonl"
        names = list(dict.fromkeys(names))  # dedupe, preserve order
        if not names:
            print(f"[warn] no entities found for domain={domain} ({source})")
            continue
        lexicon[domain] = {e: epi.transliterate(e) for e in names}
        print(f"[lexicon] {domain}: {len(names)} entities from {source}")
    return lexicon


def entity_ipa_for(domain: str) -> Dict[str, str]:
    """A single domain, or every domain merged when domain == 'all'."""
    if domain == "all":
        merged: Dict[str, str] = {}
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
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception:
        try:  # webm/opus, mp4/aac, mp3, ...
            return decode_with_av(raw)
        except Exception as exc:
            raise HTTPException(400, f"could not decode audio '{filename}': {exc}")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != TARGET_SR:
        audio = soxr.resample(audio, sr, TARGET_SR).astype(np.float32)
    return audio, sr


# --------------------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------------------

app = FastAPI(title="wake2adapt serving API")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "asr_loaded": ASR is not None,
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
    items = [{"entity": e, "ipa": ipa} for e, ipa in list(table.items())[:limit]]
    return {"domain": domain, "size": len(table), "items": items}


@app.post("/retrieve")
def retrieve(text: str = Form(...), domain: str = Form("roads"), top_k: Optional[int] = Form(None)):
    """Retriever only -- feed it a transcription directly, no GPU involved."""
    k = top_k or CFG.top_k
    t0 = time.perf_counter()
    hits, text_ipa = retrieve_top_k(text, entity_ipa_for(domain), k=k)
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
):
    """ASR (zero-shot, or 1-shot adaptation when ref_audio is sent) + phonetic retrieval."""
    if ASR is None:
        raise HTTPException(503, "server started with --no-asr; only /retrieve is available")
    k = top_k or CFG.top_k
    entity_ipa = entity_ipa_for(domain)

    t0 = time.perf_counter()
    wav, orig_sr = decode_audio(await audio.read(), audio.filename or "audio")
    ref_wav = None
    if ref_audio is not None and ref_audio.filename:
        ref_wav, _ = decode_audio(await ref_audio.read(), ref_audio.filename)
    t_decode = time.perf_counter()

    asr_result = ASR.run_asr(audio=wav, ref_audio=ref_wav, reference_text=ref_text)
    t_asr = time.perf_counter()

    hits, asr_ipa = retrieve_top_k(asr_result, entity_ipa, k=k)
    t_end = time.perf_counter()

    return {
        "asr_result": asr_result,
        "asr_ipa": asr_ipa,
        "asr_adaptation": ref_wav is not None,
        "ref_text": ref_text,
        "domain": domain,
        "top_k": k,
        "lexicon_size": len(entity_ipa),
        "retrieved": hits,
        "retr_entities": [h["entity"] for h in hits],
        "audio": {
            "filename": audio.filename,
            "orig_sample_rate": orig_sr,
            "seconds": round(len(wav) / TARGET_SR, 2),
        },
        "timing": {
            "decode_s": round(t_decode - t0, 3),
            "asr_s": round(t_asr - t_decode, 3),
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
    default_repo = Path(__file__).resolve().parent / "wake2adapt"

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--repo_root", default=os.environ.get("W2A_REPO", str(default_repo)),
                   help="wake2adapt checkout (default: ./wake2adapt next to this file)")
    p.add_argument("--jsonl_dir", default=None, help="default: {repo_root}/L2-KPNS-jsonl")
    p.add_argument("--lexicon_csv_dir", default=None,
                   help="directory holding the L2-KPNS {domain}_200.csv files; "
                        "falls back to the answers in the jsonl files")
    p.add_argument("--domains", default=",".join(DOMAINS))
    p.add_argument("--model_path", default=os.environ.get("W2A_MODEL", "Qwen/Qwen2.5-Omni-7B"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max_new_tokens", type=int, default=256)
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

    CFG = ServerConfig(args)
    LEXICON = load_lexicon(CFG)
    if not CFG.no_asr:
        ASR = load_asr(CFG)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
