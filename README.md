# w2a-api

Minimal HTTP serving wrapper around [`wake2adapt`](https://github.com/solee0022/wake2adapt):
POST an audio file, get back
the Qwen2.5-Omni transcription plus the phonetic edit-distance retrieval hits over the L2-KPNS
entity lexicon (the same `retrieve_top_k` scoring the offline scripts use).

## Setup

The ASR pipeline and the L2-KPNS jsonl data live in a separate repository, and the browser demo in
another; both are cloned next to `server.py` and are not vendored here:

```bash
git clone https://github.com/21jun/w2a-api.git && cd w2a-api
git clone https://github.com/solee0022/wake2adapt.git   # ASR pipeline + L2-KPNS-jsonl (required)
git clone https://github.com/21jun/w2a-demo.git         # browser demo (optional)

uv venv --python 3.11 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt
export HF_HOME=/path/with/room/for/21GB   # on the KAIST box: source ../env.sh
```

`--repo_root` points somewhere else if the `wake2adapt` checkout is not a sibling of `server.py`.

## Run

On Windows with an NVIDIA GPU, use a separate Windows environment. The existing `.venv`
may be a Linux checkout and is left intact:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup-windows.ps1
.\.venv-win\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\start-study.cmd
```

To stop the API and its GPU worker, run `.\end-study.cmd`. To preview the
processes and cache directories affected, run `.\end-study.cmd -WhatIf`.
This removes Python bytecode caches and `.cache/wheels`, while preserving
the downloaded model in `.cache/huggingface`, server logs, recordings, and study data.
The frontend has its own `w2a-user-study\end-study.cmd`; run both commands
when ending the complete local study environment.

`setup-windows.ps1` installs CUDA 12.6 PyTorch 2.8 / torchvision 0.23 / torchaudio 2.8,
Transformers 4.57.6, and the tested versions in `requirements-windows.lock.txt`
(falling back to `requirements-windows.txt` if the lock file is missing), then checks
GPU allocation, then prefetches the model (use `-SkipModelDownload` to skip that step).
`start-study.cmd` uses this project Python explicitly, sets UTF-8, and
caches model files in `.cache/huggingface` unless `HF_HOME` is already configured.
The default is local port 8000, int8 quantization, bfloat16, and GPU `cuda:0`.
The launcher refuses a second project instance even while the first is still loading.
Reading the full lexicons and generating their IPA can take several minutes before
the API starts listening; model downloads are cached separately.
For a different port or lower-memory quantization, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start-study.ps1 -Port 8000 -Quant nf4
```

The first model download can exceed the worker's startup deadline. Prefetch the checkpoint
before launching in that case:

```powershell
$env:HF_HOME = Join-Path $PWD '.cache\huggingface'
.\.venv-win\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-Omni-7B')"
```

```bash
.venv/bin/python server.py --port 8000                  # loads Qwen2.5-Omni-7B on cuda:0
.venv/bin/python server.py --port 8000 --no-asr         # retriever only, no GPU
.venv/bin/python server.py --port 8000 --device cuda:3 --top_k 10
```

Useful flags: `--model_path` (default `Qwen/Qwen2.5-Omni-7B`, also `W2A_MODEL`), `--device`,
`--dtype`, `--quant` (see below), `--max_new_tokens`, `--domains`, `--top_k`, `--repo_root`
(default `./wake2adapt`), `--max_transcript_chars` (default 20; longer transcripts skip DB
search, see below),
`--lexicon_csv_dir` (point it at `L2-KPNS/metadata/recording_targets` to use the official
`{domain}_200.csv` lexicon; without it the 200 entities per domain are rebuilt from the
`answer` fields in `wake2adapt/L2-KPNS-jsonl/`).

## Audio validation and inference deadlines

The target and optional reference must decode successfully to a nonempty, finite mono
waveform before ASR starts. Empty uploads, undecodable/unsupported files, zero-sample audio,
and NaN/Inf samples return HTTP 400 with `detail.code: "INVALID_AUDIO"`, a user-facing
`detail.message`, and the failing `detail.filename`. Soundfile falls back to PyAV for browser
formats. A decoder may accept partially damaged files; this is a decodability check, not a
guarantee of bit-for-bit file integrity. Valid silence/noise is still passed to the model.

ASR runs in a persistent spawned GPU process. `--asr_timeout_s` defaults to 10 seconds per
call, excluding upload, decoding, model loading, and retrieval. On expiry, the process is
terminated (with a kill fallback) and HTTP 504 / `ASR_TIMEOUT` is returned; process cleanup
can add up to approximately two seconds. The failed audio is not automatically retried.
The model reloads in the background. Calls during another inference or reload return
HTTP 503 / `ASR_BUSY`; an unavailable model returns `ASR_UNAVAILABLE`. Worker failures
return `ASR_WORKER_FAILED` and also trigger reload. Startup/reload has no time limit, so the
first run can download the model weights; failed recovery leaves ASR unavailable until the
server is restarted. `/health.asr_loaded` reports worker readiness.

```bash
python server.py --asr_timeout_s 10
python -X utf8 -m unittest discover -s tests -v
```

The browser STT proxies preserve the new structured error messages, codes, and HTTP statuses.
The 10-second value is provisional, not derived from measured production percentiles.
Output validation, hallucination detection, and re-transcription are not implemented.
The ASR no longer removes a matching reference-text prefix from the transcript; it still
strips the model's `Transcription:` formatting prefix.

`POST /cancel` accepts `request_id` and signals cancellation for that ID. ASR polls the
signal every 50 ms; cancellation terminates its GPU worker and reloads the model, with
the interrupted request returning HTTP 499 / `REQUEST_CANCELLED`. This response code
also applies to cancelled retrieval. The cancel response's `was_running` indicates
whether an active stage was found, not whether worker cleanup has already completed.
Cancellation received before upload finishes or between ASR and retrieval is remembered
for five minutes (bounded to 4096 IDs). During decoding, cancellation is checked as soon
as decoding returns, before the model is called. Use a new ID per analysis and reuse it
for that analysis's ASR and retrieval stages. IDs are coordinated within one API process.

`POST /transcribe` now accepts `request_id` and `with_retrieval=false` (ASR only), and
`POST /retrieve` accepts the same `request_id`. The study UI forwards the ID to both
stages, avoiding duplicate retrieval after ASR. Its cancel action now reaches either stage.

DB search uses `--retrieval_timeout_s` (default 20 seconds), for both `/retrieve` and
retrieval inside `/transcribe`. It checks cancellation and a monotonic deadline before
and after transliteration, between entries, and after each edit-distance calculation.
On expiry it stops searching and returns HTTP 504 / `RETRIEVAL_TIMEOUT`, without partial
results. An individual transliteration/distance operation must finish before a checkpoint
can interrupt it. Search cancellation/timeout does
not restart the GPU model. Empty or whitespace-only transcripts are rejected before
phonetic search with HTTP 400 / `EMPTY_TRANSCRIPT`. Transcripts of `--max_transcript_chars`
(default 20) characters or more, after trimming surrounding whitespace, are rejected the
same way with HTTP 400 / `TRANSCRIPT_TOO_LONG`; inner spaces count toward the length. The
check runs for both `/retrieve` and retrieval inside `/transcribe`, so with
`with_retrieval=false` the long ASR text is still returned and the follow-up `/retrieve`
fails. The `/transcribe` error response does not include `asr_result`.

```json
{"detail": {"code": "TRANSCRIPT_TOO_LONG",
            "message": "DB 검색 실패: ASR 결과가 23자로 너무 깁니다 (20자 미만만 검색합니다)."}}
```

```bash
python server.py --max_transcript_chars 20
```

The study UI does not issue a DB request for an empty ASR result: it immediately shows
a retrieval failure. Nonempty transcripts get at most two retrieval requests, using the
same ASR text; cancellation prevents retry. Each server search has a 20-second default
deadline, so two timed-out searches take approximately 40 seconds plus transport overhead.
After both requests fail, the UI displays a retrieval error and keeps the existing audio
and transcript rather than counting it as a failed pronunciation or restarting ASR.

Trial logging runs independently from the analysis result and target progression. Each
bundle contains the original row and audio blobs, is placed in a browser IndexedDB outbox,
and is sent up to twice per delivery pass. Failed bundles remain queued and retry every
60 seconds, when connectivity returns, on page reload, or via the retry button. The server
deduplicates by session/target/attempt, including when a prior write succeeded but its
response was lost. Logging failure never requires re-recording. If browser storage is
unavailable/full, bundles remain in memory and the UI warns against closing the page.
IndexedDB persistence still depends on browser storage availability and retention.

## VRAM and quantization

Serving only needs the *thinker* (LLM + audio encoder). The server therefore skips the
talker/token2wav speech stack and the vision tower by default, and `--quant` can put the LLM
weights in 8-bit or 4-bit through bitsandbytes:

```bash
.venv/bin/python server.py --port 8000 --device cuda:0                  # bf16, the default
.venv/bin/python server.py --port 8000 --device cuda:0 --quant nf4      # 4-bit, fits an 11 GB card
.venv/bin/python server.py --port 8000 --device cuda:0 --quant int8
.venv/bin/python server.py --port 8000 --dtype float16 --quant nf4      # pre-Ampere cards (no bf16)
```

Measured on this box (B200, transformers 4.57, 60 s audio = the demo's recording cap):

| `--quant` | weights | load peak | 60 s inference peak | process total | smallest card |
| --- | --- | --- | --- | --- | --- |
| `none` (bf16) | 15.39 GiB | 16.65 GiB | 16.04 GiB | 17.4 GiB | 24 GB |
| `int8` | 9.31 GiB | 10.18 GiB | 9.97 GiB | 11.4 GiB | 16 GB |
| `nf4` | 6.70 GiB | 7.30 GiB | 7.36 GiB | 8.3 GiB | 11 GB |

"Process total" is what `nvidia-smi` reports, i.e. including the ~0.7 GiB CUDA context. For
reference, loading the full checkpoint the way the offline scripts do (talker loaded, then
discarded) peaks at 20.8 GiB and settles at 21.6 GiB.

Quantization notes:

- bitsandbytes needs a GPU with compute capability >= 7.5 (Turing or newer). RTX 2080 Ti works
  with `--dtype float16`; GTX 1080 Ti (Pascal) does not.
- The audio encoder and the output head are never quantized (`llm_int8_skip_modules`): they are
  about 2 GiB together and quantizing them costs accuracy on Korean entity names.
- `nf4` transcripts drifted slightly on one of three test clips (an inserted space), but the
  retrieved entity stayed rank 1 in every case. Run `compute_metrics.py` over
  `L2-KPNS-jsonl` before trusting it for numbers you publish.
- 4-bit generation is slower per token than bf16; the win is memory, not speed.

`--keep_audio_output` loads the talker/token2wav stack (+4.2 GiB at load time) and `--keep_visual`
keeps the vision tower (+1.26 GiB). Neither is used by `/transcribe`; they exist for parity with
the offline pipeline.

## Endpoints

| method | path | what it does |
| --- | --- | --- |
| `GET` | `/` | one-page browser UI: upload audio, see ASR + the ranked retrieval table |
| `GET` | `/health` | model / device / lexicon sizes / timeouts / `max_transcript_chars` |
| `GET` | `/lexicon?domain=roads&limit=20` | peek at the entity lexicon and its IPA |
| `POST` | `/transcribe` | audio -> ASR -> top-k retrieval |
| `POST` | `/retrieve` | text -> top-k retrieval (no GPU, for debugging the retriever) |
| `POST` | `/cancel` | cancel ASR/retrieval by multipart `request_id` |

`/transcribe` is multipart form data:

| field | required | default | note |
| --- | --- | --- | --- |
| `audio` | yes | | wav/flac/mp3/...; anything soundfile or librosa reads, resampled to 16 kHz |
| `domain` | no | `roads` | `roads` / `content` / `restaurants` / `stations` / `all` |
| `top_k` | no | 10 | |
| `ref_audio` | no | | 1-shot reference from the same speaker -> turns on ASR adaptation |
| `ref_text` | no | `""` | the reference transcription that goes with `ref_audio` |
| `request_id` | no | generated | cancellation ID, 1–64 word characters or hyphens |
| `with_retrieval` | no | `true` | `false` returns ASR only; call `/retrieve` separately |

```bash
# zero-shot
curl -s -X POST localhost:8000/transcribe -F audio=@RD21899_P001.wav -F domain=roads | jq

# 1-shot ASR adaptation
curl -s -X POST localhost:8000/transcribe \
     -F audio=@RD21899_P001.wav -F domain=roads \
     -F ref_audio=@RD16245_P001.wav -F ref_text=상곡안길 | jq

# retriever only
curl -s -X POST localhost:8000/retrieve -F text=상국안길 -F domain=roads -F top_k=5 | jq
```

Response:

```json
{
  "asr_result": "상국안길",
  "asr_ipa": "saŋkukankil",
  "asr_adaptation": true,
  "domain": "roads",
  "top_k": 5,
  "lexicon_size": 200,
  "retrieved": [
    {"rank": 1, "entity": "상곡안길", "score": 0.91, "distance": 1, "ipa": "saŋkokankil"},
    {"rank": 2, "entity": "상고론길", "score": 0.73, "distance": 3, "ipa": "saŋkoronkil"}
  ],
  "retr_entities": ["상곡안길", "상고론길"],
  "audio": {"filename": "RD21899_P001.wav", "orig_sample_rate": 16000, "seconds": 1.8},
  "timing": {"decode_s": 0.01, "asr_s": 0.62, "retrieval_s": 0.14, "total_s": 0.77}
}
```

`retr_entities` is the same list the offline pipeline writes into its result jsonl, so the
results are directly comparable with `compute_metrics.py`.

## Demo UI (`w2a-demo`)

The Next.js demo in `w2a-demo/` talks to this API. Record a greeting once (it becomes the 1-shot
reference), then practice words; each practice upload is proxied by `w2a-demo/app/api/stt/route.ts`
to `POST /transcribe`, and the page shows the transcription, its IPA and the ranked retrieval table.

```bash
# terminal 1
.venv/bin/python server.py --port 8000 --device cuda:0

# terminal 2 (Node 22.13+; see w2a-demo/README.md)
cd w2a-demo
printf 'W2A_API_URL=http://127.0.0.1:8000\nW2A_DOMAIN=roads\nW2A_TOP_K=10\n' > .env
npm ci && npm run dev
```

Leave `W2A_API_URL` unset and the demo keeps its standalone behaviour (HTTP 202,
`adaptationStatus: "not_configured"`) instead of inventing a transcript.

## Notes

- The ASR pipeline itself is not reimplemented: `server.py` loads
  `wake2adapt/src/infer_qwen2.5_omni.py` by path (its filename has a `.` in it) and calls
  `Qwen25OmniASRPipeline.run_asr`, so prompt construction stays in one place.
- One request at a time per process — generation is not batched. For load, run several
  processes on different GPUs behind a proxy, or move to `infer_qwen2.5_omni_batch.py`.
