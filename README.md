# Omni-Indic Voice AI — Consolidated Monorepo

A low-cost, self-hosted **speech-to-speech AI for Indian regional languages** (Tamil, Telugu, Kannada, Hindi + English), with safety-aware acoustic analysis for emergency/distress detection.

This repository consolidates the full body of work behind the project — from the earliest working prototype, through a production/paper-grade pipeline, an experimental unified-transformer research track, and the custom Tamil TTS voice we trained ourselves.

> **What this repo is.** A consolidated, organized snapshot of a multi-part team project, assembled for portfolio/review purposes. Each layer below was originally its own repository; here they live side by side with a map of how they connect. See [Team & Attribution](#team--attribution).

---

## The four layers

| # | Layer | What it does | Maturity |
|---|-------|--------------|----------|
| `01` | **Demo-THIT prototype** | First end-to-end **working** system: a FastAPI voice-call app with an 8-stage pipeline (VAD → Whisper STT → language/script detection → safety-signal extraction → safety gate → policy → Llama-3.1-8B → Edge TTS), Redis sessions, RAG retrieval, admin panel, and a web UI. Healthcare-oriented (doctor/department/FAQ config). | ✅ Working prototype |
| `02` | **voice-model-ai** | Production / research-paper pipeline: LID (MMS-LID) → Whisper ASR → Ollama (llama3.2) with semantic cache (FAISS) + Redis context → Piper TTS, plus a **parallel Mimi-encoder emergency classifier**. WebRTC + Twilio ingress. Benchmarked on an AMD MI300X GPU. | ✅ Core verified |
| `03` | **apollo-engine** | Experimental **de-novo unified transformer** track: extends Sarvam-1 2B with SNAC audio tokens for a single model that "hears" and "speaks." Architecture is implemented; the model is **not yet trained**, so it is a research scaffold, not a runnable system. | 🔬 Research scaffold |
| `04` | **tts-training** | The pipeline that produced our own **Tamil Piper voice** (`ta_IN-iitm-female-s1-medium.onnx`, used by layers 01/02): dataset download/extraction scripts (IndicTTS_Tamil), Coqui-VITS and Piper training notebooks, and the trained voice assets. | ✅ Produced a real voice |

---

## How the layers connect

```
                          ┌─────────────────────────────┐
   04 tts-training  ─────▶│  Trained Tamil voice (.onnx) │─────┐
   (train the voice)      └─────────────────────────────┘     │ used by
                                                               ▼
   01 Demo-THIT prototype  ──(matured / rewritten into)──▶  02 voice-model-ai
   (first working system,                                   (production pipeline
    safety gate + RAG + UI)                                  + emergency classifier,
                                                              GPU-benchmarked)

   03 apollo-engine  ── separate research direction: one unified model instead of a cascade
```

- **01 → 02**: The prototype proved the 8-stage cascade end to end. The production layer is the cleaner, benchmarked rewrite of that idea. The prototype's **text safety gate and RAG grounding have now been folded into 02** (`src/llm/safety_gate.py`, `src/llm/rag_retriever.py`; see `02-voice-model-ai/INTEGRATION_SAFETY_RAG.md`). The admin/healthcare config and web UI still live only in **01**.
- **04 → 01/02**: Both runtime pipelines speak Tamil using the voice trained in layer 04.
- **03**: An independent bet on a single unified speech-to-speech transformer (vs. the cascade). Promising architecture, not yet trained.

---

## Verified component status (be precise when citing this)

| Component | Status | Evidence |
|-----------|--------|----------|
| LLM orchestration (Ollama, semantic cache, Redis sessions, fillers) | ✅ Verified working | `02-voice-model-ai/test_output.txt` — real run, pipeline completed in 0.21s |
| Text safety gate + RAG grounding (merged from layer 01) | ✅ Verified working | `02-voice-model-ai/test_safety_rag.py` — 14/14 pass, real Ollama end-to-end |
| Custom Tamil Piper TTS voice | ✅ Real trained asset | `02-voice-model-ai/ta_IN-iitm-female-s1-medium.onnx` (61 MB) |
| 8-stage prototype (VAD→…→TTS) with safety gate + RAG + UI | ✅ Working prototype | `01-demo-thit-prototype/app.py` (FastAPI, ~45 KB) |
| GPU latency benchmarks | ✅ Backed by artifacts | `02-voice-model-ai/results/` (measured on AMD MI300X, 2026-02-22) |
| WebRTC / Twilio ingress | 🟡 Coded, needs full GPU stack to exercise | `02-voice-model-ai/src/realtime/` |
| Emergency detector (Mimi + classifier) | 🟡 Code + benchmarks exist; **trained weights not in this repo** | `02-voice-model-ai/src/train_emergency_classifier.py` |
| Unified SNAC + Sarvam transformer | 🔬 Architecture only, **untrained** | `03-apollo-engine/.../models/audio_llm.py` |

**Measured performance** (from the paper, on a rented AMD MI300X — *not* on a laptop): time-to-first-sound ≈ 488 ms, end-to-end ≈ 742 ms, direct operating cost ≈ ₹0.95/min. Full breakdown: `02-voice-model-ai/results/latency_validation_report.md`.

---

## Repository layout

```
.
├── 01-demo-thit-prototype/   # FastAPI 8-stage voice-call app (working prototype)
├── 02-voice-model-ai/        # Production pipeline + emergency classifier + benchmarks
├── 03-apollo-engine/         # Unified SNAC + Sarvam-1 research scaffold
├── 04-tts-training/          # Tamil voice training: notebooks, data scripts, voice assets
├── .gitignore
└── README.md                 # (this file)
```

Each subfolder keeps its own README and setup instructions.

## Quick start (per layer)

Each layer runs independently. Copy `.env.example` → `.env` where present, then follow the layer's own README:

- **Try the working prototype:** `cd 01-demo-thit-prototype && pip install -r requirements.txt && bash run.sh`
- **Production pipeline:** see `02-voice-model-ai/README.md` (needs Ollama + Redis; GPU deps are ROCm-pinned).
- **Voice training:** open the notebooks in `04-tts-training/notebooks/` (Colab/GPU).

> **Note on dependencies:** `02-voice-model-ai/requirements.txt` pins `torch==2.10.0+rocm7.1` (AMD ROCm), because it was deployed on an AMD MI300X. On other hardware, install the matching CPU/CUDA build of PyTorch instead.

## Security / privacy

- No secrets are committed. `.env` files are git-ignored; use the provided `.env.example` templates.
- Sample audio and dataset metadata included are for demonstration only.

## Team & Attribution

This is a **team project**. The three engine codebases (layers 01–03) were primarily developed by **Jeevith G ([@jeevithg090](https://github.com/jeevithg090))**, originally in the repositories:

- `01` — https://github.com/jeevithg090/Demo-THIT
- `02` — https://github.com/jeevithg090/Voice-AI-Model
- `03` — https://github.com/jeevithg090/Speech-to-Speech

This consolidated monorepo is maintained by **Shiva Teja ([@ShivaTeja05](https://github.com/ShivaTeja05))**. Please refer to the original repositories for authoritative commit history.

## License

See individual layers for their license terms (`02-voice-model-ai/LICENSE`). Some components are marked proprietary by their original authors — check before reuse.
