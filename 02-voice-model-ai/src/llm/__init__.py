"""
Layer 2 LLM Integration Package
"""
from .embedding_agent import process_with_context, process_query, get_rag_retriever
from .redis_context import RedisContextManager
from .safety_gate import run_safety_gate, extract_layer1_signals, build_safety_directive
from .rag_retriever import RAGRetriever, load_documents_from_json
