# Architecture — Layer by Layer

Complete architecture of the Omni-Indic Voice AI project, documenting every
component we built across the four layers. Read the top-level [README](README.md)
first for the high-level map; this file is the detailed breakdown.

**Languages supported:** Hindi, Tamil, Telugu, Kannada, English.

**One-line-per-layer:**

| Layer | Name | Role | Status |
|-------|------|------|--------|
| 01 | Demo-THIT prototype | First working end-to-end voice-call system | ✅ Working prototype |
| 02 | voice-model-ai | Production/paper pipeline (benchmarked) | ✅ Core verified |
| 03 | apollo-engine | Unified speech-to-speech transformer (research) | 🔬 Architecture only, untrained |
| 04 | tts-training | Training the custom Tamil voice used by 01/02 | ✅ Produced a real voice |

---

## Layer 01 — Demo-THIT prototype

**What it is:** our first fully working real-time voice assistant — a single
FastAPI service implementing an 8-stage pipeline with a browser UI and an admin
panel. Healthcare-oriented (hospital FAQ / doctor / department knowledge).

**Stack:** FastAPI · Faster-Whisper · Llama-3.1-8B (4-bit) · Edge TTS · Redis ·
sentence-transformers + FAISS (RAG) · vanilla JS frontend.

### The 8-stage pipeline (`app.py`)

```
mic audio (browser)
   │
1. VAD                  run_vad()               energy/webrtc voice-activity detection
2. STT                  run_stt()               Faster-Whisper (large-v3) transcription
3. Language/script      detect_script_language()resolves spoken language vs script
4. Layer-1 signals      extract_layer1_signals()intent / urgency / tone / stress
5. Safety gate          run_safety_gate()       5 escalation rules → should_escalate
6. Policy               apply_policy()           hospital policy: tone, max words, disclaimers
7. LLM                  generate_llm_response() Llama-3.1-8B grounded in RAG context
8. TTS                  generate_tts()           Edge TTS → audio back to caller
```

### Components

| Path | Purpose |
|------|---------|
| `app.py` | The whole pipeline + FastAPI server (~1200 lines): model loading, the 8 stages, session + admin REST API. |
| `models/embeddings.py` | RAG: `EmbeddingManager` (sentence-transformers) + `RAGRetriever` (FAISS index, numpy cosine fallback), `load_documents_from_json/redis`. |
| `models/redis_store.py` | `RedisStore`: hospital config, doctors, departments, FAQs, and knowledge documents. |
| `models/conversation.py` | `ConversationManager`: session history, repeated-query detection (feeds safety rule 5). |
| `static/index.html`, `app.js`, `styles.css` | Patient-facing web UI (record → converse). |
| `static/admin.html`, `admin.js`, `admin.css` | Admin panel: live-edit hospital config, doctors, departments, FAQs. |
| `data/sample_hospital.json` | Seed knowledge base. |
| `scripts/seed_data.py` | Loads seed data into Redis. |
| `docs/ARCHITECTURE.md` | The prototype's own in-depth design doc. |

### REST API (selected)
- `POST /process` — main turn: audio in → spoken response out
- `POST /api/session`, `GET /api/session/{id}` — session management
- `GET/POST/PUT/DELETE /api/admin/{config,doctors,departments,faqs}` — admin CRUD
- `GET /`, `GET /admin`, `GET /health`, `GET /languages`

---

## Layer 02 — voice-model-ai (production pipeline)

**What it is:** the cleaner, benchmarked rewrite of the prototype as modular,
swappable services, with real-time transport (WebRTC + telephony), a **parallel
acoustic safety path**, an admin/agent runtime, and measured GPU performance.
This is the layer described in the research paper.

**Stack:** Python · WebRTC (aiortc) + Twilio Media Streams · MMS-LID · Whisper ·
Ollama (llama3.2) · FAISS semantic cache · Redis · Mimi encoder + emergency
classifier · Piper + Edge TTS · SQLite admin store · Flask/gunicorn.

### Runtime pipeline

```
WebRTC / Twilio ─▶ normalize + endpoint (energy VAD)
                        │
        ┌───────────────┼────────────────────────────┐
        ▼               ▼                              ▼
   MMS-LID          Mimi encoder +                Whisper ASR
   (language)       distress classifier           (transcript)
        │           (parallel safety)                  │
        │                                              ▼
        │                                   text safety gate + RAG
        └──────────────▶ LLM orchestration (Ollama, per-language prompt,
                          semantic cache + Redis context, fillers)
                                   │
                          stream chunker (4-token)
                                   │
                          per-language TTS (Piper / Edge)
                                   │
                          audio out  +  persist turns/recordings (SQLite)
```

### Components by subsystem

**Audio transport — `src/realtime/`**
| File | Purpose |
|------|---------|
| `webrtc_server.py`, `signaling_server.py` | WebRTC offer/answer + streaming server. |
| `twilio_track.py` | Twilio Media Streams (telephony) ingress. |
| `audio_track.py` | Audio frame handling / endpointing. |
| `lid_only_server.py` | Standalone language-ID server. |
| `client.html`, `call_interface.html`, `push_to_talk.html`, `admin_panel.html`, `lid_test.html` | Browser frontends: full call UI, push-to-talk, admin panel, LID test harness. |

**LLM orchestration — `src/llm/`**
| File | Purpose |
|------|---------|
| `embedding_agent.py` | Main orchestrator `process_with_context()`: safety → RAG → cache → context → LLM → store. |
| `ollama_client.py` | Ollama chat/emb, streaming + non-streaming. |
| `semantic_cache.py` | FAISS semantic cache (skip LLM on near-duplicate turns). |
| `redis_context.py` | Redis-backed multi-turn session memory. |
| `filler_manager.py` | Natural fillers / back-channels ("let me check…") to mask LLM latency. |
| `stream_chunker.py` | Chunks the LLM token stream (4-token default) for low-latency TTS. |
| `similarity.py` | Embedding similarity helpers. |
| `safety_gate.py` | **Merged from layer 01** — text safety gate (5 escalation rules). |
| `rag_retriever.py` | **Merged from layer 01** — RAG grounding via Ollama embeddings. |

**TTS — `src/tts/`**
| File | Purpose |
|------|---------|
| `tts_manager.py` | Per-language routing: Piper ONNX for `en/hi/ta/te`, Edge TTS for `kn` + fallback. |
| `edge_tts.py` | Microsoft Edge TTS backend. |

**Safety / voice analysis (acoustic)**
| File | Purpose |
|------|---------|
| `src/train_emergency_classifier.py`, `src/train/train_emergency_embedding.py` | Train the distress/emergency classifier on Mimi embeddings. |
| `src/test_emergency_classifier.py`, `src/test_single_audio.py` | Inference/eval. |
| `src/benchmark_mimi_latency.py`, `src/benchmark_emergency_latency.py` | Latency benchmarks. |

**Admin / agent runtime — `src/admin/`**
| File | Purpose |
|------|---------|
| `store.py` | `AdminStore` (SQLite): agents, prompt templates, config, and persisted turns + audio recordings. |
| `runtime.py` | Resolves the active **agent profile**, renders per-domain prompt templates, and encodes the human-handover directive (`HANDOVER_MODE`, default `phone`). |

**Data prep — `src/utils/`, `src/datasets/`**
- `process_ravdess.py`, `prepare_ravdess_emergency.py`, `add_neutral_ravdess_to_normal.py` — emergency dataset from RAVDESS.
- `process_commonvoice_indic.py`, `prepare_normal_clips.py` — Indic "normal" speech.
- `encode_with_mimi.py` — encode clips to Mimi embeddings.
- `build_metadata.py`, `create_splits*.py` — metadata + train/val/test splits.
- `datasets/speech_dataset.py` — PyTorch dataset.

### Key configuration (`src/config.py`)
`LLM_MODEL=llama3.2:3b` · `SUMMARIZE_MODEL=llama3.2:1b` · `EMBED_MODEL=nomic-embed-text`
· `WHISPER_MODEL=openai/whisper-small` · `LID_MODEL=facebook/mms-lid-256`
· `ENABLE_TWILIO` · `ENABLE_LID_AUTODETECT` · `HANDOVER_MODE`.

### Measured performance (paper, AMD MI300X, 2026-02-22)
TTFS ≈ 488 ms · end-to-end ≈ 742 ms · direct cost ≈ ₹0.95/min. Per-stage table
and artifacts in `02-voice-model-ai/results/` and `README.md`.

---

## Layer 03 — apollo-engine (unified transformer, research)

**What it is:** an independent research direction — instead of a cascade, a
**single unified model** that both understands and speaks, by extending an LLM
with audio tokens. Architecture is implemented; **the model is not trained**, so
this is a scaffold/prototype, not a runnable system.

**Stack:** PyTorch · Sarvam-1 2B · SNAC neural audio codec · webrtcvad · PEFT.

### Intended flow

```
Audio In ─▶ SNAC encoder ─▶ [ Sarvam-1 2B + audio-token vocabulary ] ─▶ SNAC decoder ─▶ Audio Out
                              (single next-token model: "hears" and "speaks")
```

### Components — `apollo_voice_engine/`
| File | Purpose |
|------|---------|
| `models/audio_llm.py` | `AudioLLM`: extends Sarvam-1 with +4096 SNAC audio tokens for unified text+audio next-token prediction. |
| `models/snac_wrapper.py` | `SNACWrapper`: encode/decode audio ↔ discrete SNAC codes. |
| `models/speaker_encoder.py` | Speaker embedding (voice identity / cloning). |
| `safety/classifier.py` | Safety classifier for the unified path. |
| `inference/engine.py` | `VoiceActivityDetector`, `InferenceEngine` (chunked audio in, streamed response out), `StreamingSession`. |
| `speech_pipeline.py` | Alternative cascade config (Whisper + Sarvam + MMS-TTS) with metrics. |
| `integration/webrtc_server.py` | WebRTC integration (currently a mock). |
| `config/model_config.yaml` | Model/architecture config. |

### Scripts & notebooks
- `scripts/`: `train_audio_llm.py`, `verify_architecture.py`, `verify_quick.py`, `unified_demo.py`, `unified_cli_test.py`, `piper_tts_server.py`, `download_models.py`, `api_demo.py`, `test_snac_shape.py`, `test_anchoring_simulation.py`.
- `notebooks/`: unified-model training (`train_audio_llm`, `_v2`), `apollo_voice_training`, `apollo_voice_optimization`, `apollo_voice_phase4_safety`, and several speech-to-speech demos.
- `demo/index.html`, `demo/piper_demo.html` — demo frontends.

> **Honesty note:** the README's "unified de-novo transformer" describes this
> intended design. Because no trained weights exist, running it produces noise —
> present it as *designed/prototyped*, not *working*.

---

## Layer 04 — tts-training (the Tamil voice)

**What it is:** the pipeline that produced our **own** Tamil TTS voice
(`ta_IN-iitm-female-s1-medium.onnx`), which layers 01/02 load as the Tamil entry
in their per-language TTS set. Built from the SPRINGLab **IndicTTS_Tamil**
dataset.

**Stack:** Piper (rhasspy) + piper-phonemize · Coqui VITS · espeak-ng · HTS.

### Data preparation — `data_scripts/`
| File | Purpose |
|------|---------|
| `download_dataset.py`, `retry_downloads.py`, `fix_corrupt_file.py` | Fetch IndicTTS_Tamil parquet shards from Hugging Face. |
| `inspect_parquet.py`, `check_gender.py` | Inspect shards / confirm speaker gender. |
| `extract_dataset.py`, `extract_fixed_part.py` | Extract WAV + text from parquet. |
| `prepare_tamil_data.py` | Build Piper dataset (`metadata.csv` + `wav_22050/`). |
| `manual_link_dataset.py` | Link dataset into the training dojo. |
| `download_checkpoint.py` | Fetch a pretrained Piper checkpoint to fine-tune from. |
| `train_piper.sh` | Piper training launcher. |

### Training notebooks — `notebooks/`
- `Piper_Tamil_Training.ipynb`, `Tamil_Piper_Official_Fix.ipynb`, `Piper_Colab_Conda_Py310.ipynb`, `piper_multilingual_training_notebook.ipynb` — Piper training (Colab/GPU, dependency fixes).
- `Coqui_VITS_Tamil.ipynb` — alternative Coqui VITS training track.

### Outputs
- `voice_assets/iitm_unified_tamil_female.htsvoice` — HTS voice asset.
- `02-voice-model-ai/ta_IN-iitm-female-s1-medium.onnx` (+ `.json`) — the trained Piper voice used at runtime.
- `Local_Mac_Training_Guide.md` — local training notes.

> Third-party tool clones used during training (`piper/`, `piper-phonemize/`,
> `TextyMcSpeechy/`) are **not** vendored here — they are upstream dependencies.

---

## How the layers relate

- **04 → 01/02:** layer 04 trained the **Tamil** Piper voice; the runtime
  pipelines load it as one of several per-language TTS voices.
- **01 → 02:** the prototype proved the cascade; layer 02 is the modular,
  benchmarked rewrite. The prototype's **safety gate + RAG** are now merged into
  layer 02 (`src/llm/safety_gate.py`, `rag_retriever.py`).
- **03:** a separate bet on one unified model instead of the cascade.

## What is per-language vs shared (multilingual strategy)

| Stage | Per-language? | Model(s) |
|-------|---------------|----------|
| Language ID | shared | MMS-LID (`facebook/mms-lid-256`) |
| ASR | shared | Whisper-small |
| LLM | shared model, per-language **prompt** | `llama3.2:3b` |
| **TTS** | **yes — different model per language** | Piper `en_US-lessac`, `hi_IN-rohan`, **`ta_IN-iitm-female` (layer 04)**, `te_IN-maya`; Edge TTS for `kn` + fallback |
