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

```bash
.venv/bin/python server.py --port 8000                  # loads Qwen2.5-Omni-7B on cuda:0
.venv/bin/python server.py --port 8000 --no-asr         # retriever only, no GPU
.venv/bin/python server.py --port 8000 --device cuda:3 --top_k 10
```

Useful flags: `--model_path` (default `Qwen/Qwen2.5-Omni-7B`, also `W2A_MODEL`), `--device`,
`--dtype`, `--quant` (see below), `--max_new_tokens`, `--domains`, `--top_k`, `--repo_root`
(default `./wake2adapt`),
`--lexicon_csv_dir` (point it at `L2-KPNS/metadata/recording_targets` to use the official
`{domain}_200.csv` lexicon; without it the 200 entities per domain are rebuilt from the
`answer` fields in `wake2adapt/L2-KPNS-jsonl/`).

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
| `GET` | `/health` | model / device / lexicon sizes |
| `GET` | `/lexicon?domain=roads&limit=20` | peek at the entity lexicon and its IPA |
| `POST` | `/transcribe` | audio -> ASR -> top-k retrieval |
| `POST` | `/retrieve` | text -> top-k retrieval (no GPU, for debugging the retriever) |

`/transcribe` is multipart form data:

| field | required | default | note |
| --- | --- | --- | --- |
| `audio` | yes | | wav/flac/mp3/...; anything soundfile or librosa reads, resampled to 16 kHz |
| `domain` | no | `roads` | `roads` / `content` / `restaurants` / `stations` / `all` |
| `top_k` | no | 10 | |
| `ref_audio` | no | | 1-shot reference from the same speaker -> turns on ASR adaptation |
| `ref_text` | no | `""` | the reference transcription that goes with `ref_audio` |

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
