# Safety Gate + RAG Integration

This document describes the text-side **safety gate** and **RAG grounding**
that were merged in from the Demo-THIT prototype (layer `01`) into this
pipeline's LLM layer (`src/llm/`).

## What was added

| File | Purpose |
|------|---------|
| `src/llm/safety_gate.py` | Text-based safety gate: extracts intent/urgency/stress signals from the ASR transcript and applies 5 escalation rules. Complements the *acoustic* emergency classifier. |
| `src/llm/rag_retriever.py` | Retrieval-Augmented Generation. Embeds a knowledge base with the existing Ollama `nomic-embed-text` (no new dependency), FAISS index with NumPy cosine fallback. |
| `data/knowledge_base.json` | Sample hospital knowledge base (appointments, departments, OPD timings, emergency numbers) in English + Tamil. |
| `test_safety_rag.py` | Integration test covering all three layers. |

## How it wires into the pipeline

`src/llm/embedding_agent.py :: process_with_context()` now runs, per turn:

```
transcript
   │
   ├─▶ safety_gate.run_safety_gate()   → if escalate, inject SAFETY OVERRIDE
   │                                      directive + bypass semantic cache
   ├─▶ rag_retriever.build_context_block() → inject KNOWLEDGE BASE grounding
   │
   ▼
semantic cache → Redis context → LLM → store
```

Two new flags on `process_with_context(...)`: `enable_safety=True`,
`enable_rag=True`. The response dict gains `escalate`, `safety`, and
`rag_used` fields.

### Safety escalation rules
1. Critical symptom present (e.g. "chest pain", "can't breathe")
2. Chest/breathing concern + medical intent
3. Low-confidence medical query (needs clarification)
4. High urgency (≥2 urgency keywords)
5. Repeated query (caller supplies `repeat_count`)

On escalation the LLM is instructed to acknowledge calmly, surface emergency
numbers (108/102/100), and offer human handover; the semantic cache is
bypassed so an emergency turn never gets a stale casual answer.

## Verification

```
$ python test_safety_rag.py
...
RESULT: 14 passed, 0 failed
```

Tested end-to-end on a machine with Ollama (`nomic-embed-text` + `llama3.2`)
and Redis running. The safety gate is pure Python and runs anywhere; RAG and
the end-to-end test require Ollama.

## Not ported (still only in layer 01)
- Admin panel / healthcare config (doctor/department/FAQ CRUD)
- Web UI

These depend on the prototype's Flask/Redis admin store and were left in
`01-demo-thit-prototype/` rather than duplicated here.
