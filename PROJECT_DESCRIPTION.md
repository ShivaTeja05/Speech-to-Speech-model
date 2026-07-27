# Omni-Indic Voice AI — Project Description

## One-line pitch
A low-cost, self-hosted **speech-to-speech AI for Indian regional languages**
(Hindi, Tamil, Telugu, Kannada, English) that holds real-time voice
conversations and detects distress/emergencies from the caller's voice — built
to make voice AI affordable enough for public-scale deployment in healthcare,
helplines, and government services.

## The problem we set out to solve
Commercial voice AI for Indian languages is priced far too high for large-scale
public use, and most systems are English-first with weak regional-language
support. We wanted to prove that a **fully self-hosted, open-source stack** could
deliver natural, real-time, multilingual voice conversations at **under ₹2 per
minute** — cheap enough for telehealth triage, emergency response, and regional
helpdesks — while adding a safety layer that most text-only systems lack:
detecting emergencies directly from the *acoustics* of a caller's voice, not
just their words.

## What it does
A caller speaks (over a browser or a phone call); the system:

1. Detects which language they're speaking,
2. Transcribes and understands them,
3. In parallel, analyzes the raw audio for distress/emergency cues,
4. Generates a grounded, safety-aware response with a large language model,
5. Speaks back in the caller's own language using a natural voice,

all in well under a second, with the whole conversation happening on
self-hosted infrastructure.

## System architecture

The project is organized into **four layers** that build on each other.

**Layer 1 — Working prototype (Demo-THIT).** The first fully working end-to-end
system: a single real-time voice-call web app implementing an **8-stage
pipeline** — voice-activity detection → speech-to-text (Faster-Whisper) →
language/script detection → intent/urgency/stress signal extraction → a
**safety gate** → policy application → LLM response (Llama-3.1-8B) →
text-to-speech (Edge TTS). It includes Redis-backed sessions,
**retrieval-augmented generation (RAG)** grounding, a healthcare admin panel
(doctors/departments/FAQs), and a browser UI. This proved the concept
end-to-end.

**Layer 2 — Production pipeline (voice-model-ai).** The prototype re-engineered
as clean, modular, swappable services with measured performance. This is the
core system:

- **Ingress:** WebRTC (browser) + Twilio Media Streams (real phone calls)
- **Language ID:** MMS-LID (`facebook/mms-lid-256`)
- **ASR:** Whisper-small
- **LLM orchestration:** Ollama running `llama3.2:3b`, with a **FAISS semantic
  cache** (skip the LLM on near-duplicate turns), **Redis** multi-turn memory,
  per-language prompting, natural **filler/back-channel** audio to mask latency,
  and a **4-token streaming chunker** for low-latency speech
- **TTS:** per-language model routing — **Piper** (local ONNX) for
  English/Hindi/Tamil/Telugu, **Edge TTS** for Kannada + fallback
- **Parallel acoustic safety:** a **Mimi neural audio encoder + a
  distress/emergency classifier** running alongside ASR, so emergencies are
  caught from *how* someone speaks, not only what they say
- **Ops:** a SQLite admin store (persists turns + recordings), configurable
  **agent profiles/templates**, and a **human-handover** mode

**Layer 3 — Unified-transformer research (apollo-engine).** An experimental,
more ambitious direction: instead of a cascade of separate models, a **single
unified transformer** that both "hears" and "speaks" — extending the Sarvam-1 2B
LLM with **SNAC neural audio tokens** so one model does STT + reasoning + TTS.
*(Honest status: the architecture is implemented but the model is not yet
trained, so this is a research scaffold, not a running system.)*

**Layer 4 — Custom TTS voice training.** We trained our **own Tamil neural
voice** (a Piper ONNX model) from the SPRINGLab IndicTTS_Tamil dataset — the
full pipeline of dataset download/extraction, preparation, and Piper/Coqui-VITS
training. This voice is the Tamil entry used by layers 1 and 2.

## Key technical contributions

1. **Low-cost production architecture** for multilingual Indic voice — fully
   self-hosted on open-source models.
2. **Parallel acoustic safety analysis** — distress/emergency detection from
   voice embeddings (Mimi), not text sentiment, so it works even when the words
   don't say "emergency."
3. **Text safety gate + RAG grounding** — a 5-rule escalation gate (critical
   symptoms, urgency, low-confidence medical queries, repeated queries) plus
   knowledge-base grounding so the assistant stays factual and escalates to
   humans when it should.
4. **Per-language TTS strategy** including a **custom-trained Tamil voice**,
   with graceful fallback.
5. **Transparent, measurement-first engineering** — real per-component latency
   and cost accounting from GPU runs.

## Results (measured on an AMD Instinct MI300X GPU)

- **Time-to-first-sound:** ~488 ms
- **End-to-end speech-to-speech latency:** ~742 ms
- **Direct operating cost:** ~₹0.95/min (target < ₹2/min) — roughly **85–90%
  cheaper** than commercial alternatives
- Per-stage breakdown (e.g., ASR ~96 ms, LLM time-to-first-token ~165 ms, TTS
  first chunk ~188 ms) is backed by saved benchmark artifacts in
  `02-voice-model-ai/results/`.

> These figures were measured on a rented AMD MI300X GPU, not a laptop.

## Tech stack
Python · WebRTC (aiortc) + Twilio · Whisper · MMS-LID · Ollama (llama3.2) ·
FAISS · Redis · Mimi encoder + PyTorch classifier · Piper + Edge TTS · Sarvam-1
2B + SNAC (research) · FastAPI/Flask · SQLite · Docker.

## Team & roles
A three-person team; all code was developed and committed on the lead's system,
so the git history is under his account.

- **Jeevith G ([@jeevithg090](https://github.com/jeevithg090)) — lead / main
  builder:** primary builder of the system; led the production pipeline, the
  unified-transformer research, and overall integration.
- **Shiva Teja ([@ShivaTeja05](https://github.com/ShivaTeja05)) — co-builder:**
  model training, multilingual integration, and training the custom Tamil voice.
- **Advita:** built the first working prototype (Layer 1).

## Honest status summary

- ✅ **Verified working:** the LLM orchestration (with semantic cache, sessions,
  safety gate, RAG — tested end-to-end), the custom Tamil voice, and the
  GPU-measured benchmarks.
- 🟡 **Built, needs full stack to exercise:** WebRTC/Twilio transport; the
  emergency classifier (code + benchmarks exist; trained weights live outside
  the repo).
- 🔬 **Research-stage:** the unified SNAC + Sarvam transformer (designed and
  scaffolded, not yet trained).

---

*See [README.md](README.md) for the high-level map and
[ARCHITECTURE.md](ARCHITECTURE.md) for the complete layer-by-layer component
breakdown.*
