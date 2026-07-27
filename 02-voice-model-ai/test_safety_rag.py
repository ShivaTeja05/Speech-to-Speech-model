"""
Integration test for the Demo-THIT -> voice-model-ai merge:
  1. Safety gate  (pure Python, always runs)
  2. RAG retrieval (real Ollama nomic-embed embeddings)
  3. End-to-end process_with_context (safety directive + RAG grounding + LLM)

Run:  python test_safety_rag.py
Requires (for parts 2/3): Ollama running with `nomic-embed-text` and an LLM.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

passed = 0
failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"   ✅ {name}")
    else:
        failed += 1
        print(f"   ❌ {name}  {detail}")


# ---------------------------------------------------------------------------
print("=" * 60)
print("1. SAFETY GATE (pure Python)")
print("=" * 60)
from src.llm.safety_gate import run_safety_gate, extract_layer1_signals

# critical symptom -> escalate
r = run_safety_gate("I have severe chest pain and cannot breathe", "en")
check("critical symptom escalates", r["should_escalate"], r)
check("  rule_1 fired", "rule_1_critical_symptom" in r["rules_triggered"], r["rules_triggered"])

# plain greeting -> no escalation
r = run_safety_gate("hello, good morning", "en")
check("greeting does not escalate", not r["should_escalate"], r)

# multiple urgency keywords -> escalate (rule 4)
r = run_safety_gate("this is an emergency, please help immediately, it is critical", "en")
check("high urgency escalates", r["should_escalate"], r)

# signals extraction sanity
s = extract_layer1_signals("I need to book an appointment with a doctor", "en")
check("intent detected = appointment", s["intent"] == "appointment", s)

# Tamil critical symptom -> escalate
r = run_safety_gate("எனக்கு நெஞ்சு வலி இருக்கிறது", "ta")
check("tamil critical symptom escalates", r["should_escalate"], r)

# ---------------------------------------------------------------------------
print("=" * 60)
print("2. RAG RETRIEVAL (real Ollama embeddings)")
print("=" * 60)
try:
    from src.llm.rag_retriever import RAGRetriever, load_documents_from_json

    kb = os.path.join(os.path.dirname(__file__), "data", "knowledge_base.json")
    docs = load_documents_from_json(kb)
    check("KB loads documents", len(docs) > 0, f"{len(docs)} docs")

    r = RAGRetriever()
    r.add_documents(docs)
    built = r.build_index()
    check("index builds (embeddings via Ollama)", built)

    if built:
        hits = r.search("how do I schedule a doctor visit?", top_k=3, language="en")
        top_cat = hits[0].get("category") if hits else None
        check("appointment query -> appointment doc on top",
              top_cat == "appointment", f"top category={top_cat}")

        hits2 = r.search("my heart hurts, chest problem", top_k=3, language="en")
        cats = [h.get("category") for h in hits2]
        check("chest query surfaces cardiology/emergency",
              any(c in ("department", "emergency") for c in cats), cats)
except Exception as e:
    check("RAG section ran", False, f"exception: {e}")

# ---------------------------------------------------------------------------
print("=" * 60)
print("3. END-TO-END process_with_context (safety + RAG + LLM)")
print("=" * 60)
try:
    from src.llm.embedding_agent import process_with_context

    # Emergency turn: expect escalate=True and an LLM answer.
    out = process_with_context(
        "I have severe chest pain and I can't breathe",
        session_id="temp_session",
        metadata={"detected_language": "en"},
        max_tokens=80,
    )
    check("pipeline returns a response", bool(out.get("response")), out.get("source"))
    check("emergency turn escalated", out.get("escalate") is True, out.get("safety"))
    print(f"      LLM said: {str(out.get('response'))[:140]}")

    # Informational turn: expect RAG used, no escalation.
    out2 = process_with_context(
        "what are the OPD timings?",
        session_id="temp_session",
        metadata={"detected_language": "en"},
        max_tokens=80,
    )
    check("info turn not escalated", out2.get("escalate") is False, out2.get("safety"))
    check("info turn used RAG grounding", out2.get("rag_used") is True, out2)
    print(f"      LLM said: {str(out2.get('response'))[:140]}")
except Exception as e:
    import traceback
    traceback.print_exc()
    check("end-to-end section ran", False, f"exception: {e}")

# ---------------------------------------------------------------------------
print("=" * 60)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 60)
sys.exit(1 if failed else 0)
