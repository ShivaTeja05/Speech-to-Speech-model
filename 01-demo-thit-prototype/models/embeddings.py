"""
Embedding and RAG System for Apollo Hospital Voice AI
Handles document embeddings and retrieval for FAQs, doctor info, etc.
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# EMBEDDING MANAGER
# ============================================================================

class EmbeddingManager:
    """Manages sentence embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = None
        self._loaded = False
    
    def load(self) -> bool:
        """Load the embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self._loaded = True
            logger.info("Embedding model loaded successfully")
            return True
        except ImportError:
            logger.warning("sentence-transformers not installed. RAG features disabled.")
            return False
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            return False
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        if not self.is_loaded:
            raise RuntimeError("Embedding model not loaded")
        return self.model.encode(texts, convert_to_numpy=True)
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.embed([text])[0]


class RAGRetriever:
    """
    Retrieval-Augmented Generation for hospital knowledge base.
    Uses FAISS for efficient similarity search.
    """
    
    def __init__(self, embedding_manager: EmbeddingManager = None):
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.index = None
        self.documents: List[Dict[str, Any]] = []
        self._index_built = False
    
    def load(self) -> bool:
        """Load embedding model"""
        return self.embedding_manager.load()
    
    @property
    def is_ready(self) -> bool:
        return self.embedding_manager.is_loaded and self._index_built
    
    def add_documents(self, documents: List[Dict[str, Any]], text_field: str = "text"):
        """
        Add documents to the retriever.
        
        Args:
            documents: List of document dicts
            text_field: Field containing text to embed
        """
        self.documents.extend(documents)
        logger.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def build_index(self) -> bool:
        """Build FAISS index from documents"""
        if not self.embedding_manager.is_loaded:
            if not self.embedding_manager.load():
                return False
        
        if not self.documents:
            logger.warning("No documents to index")
            return False
        
        try:
            import faiss
            
            # Get text from documents
            texts = []
            for doc in self.documents:
                # Combine question and answer for FAQs
                if 'question' in doc:
                    text = f"{doc.get('question', '')} {doc.get('answer', '')}"
                elif 'name' in doc and 'specialization' in doc:
                    # Doctor document
                    text = f"{doc.get('name', '')} {doc.get('specialization', '')} {doc.get('department', '')}"
                else:
                    text = doc.get('text', str(doc))
                texts.append(text)
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} documents...")
            embeddings = self.embedding_manager.embed(texts)
            
            # Build FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
            
            self._index_built = True
            logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
            return True
            
        except ImportError:
            logger.warning("faiss-cpu not installed. Using simple similarity search.")
            return self._build_simple_index()
        except Exception as e:
            logger.error(f"Error building index: {e}")
            return False
    
    def _build_simple_index(self) -> bool:
        """Fallback: store embeddings as numpy array for simple cosine search"""
        try:
            texts = []
            for doc in self.documents:
                if 'question' in doc:
                    text = f"{doc.get('question', '')} {doc.get('answer', '')}"
                elif 'name' in doc and 'specialization' in doc:
                    text = f"{doc.get('name', '')} {doc.get('specialization', '')} {doc.get('department', '')}"
                else:
                    text = doc.get('text', str(doc))
                texts.append(text)
            
            self._embeddings = self.embedding_manager.embed(texts)
            self._index_built = True
            logger.info(f"Built simple index with {len(texts)} documents")
            return True
        except Exception as e:
            logger.error(f"Error building simple index: {e}")
            return False
    
    def search(self, query: str, top_k: int = 3, language: str = "en") -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            language: Preferred language for results
            
        Returns:
            List of relevant documents with scores
        """
        if not self._index_built:
            return []
        
        if not self.embedding_manager.is_loaded:
            return []
        
        try:
            query_embedding = self.embedding_manager.embed_single(query)
            
            if self.index is not None:
                # FAISS search
                distances, indices = self.index.search(
                    query_embedding.reshape(1, -1).astype('float32'), 
                    min(top_k, len(self.documents))
                )
                
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx < len(self.documents):
                        doc = self.documents[idx].copy()
                        doc['_score'] = float(1 / (1 + distances[0][i]))  # Convert distance to similarity
                        results.append(doc)
            else:
                # Simple cosine similarity search
                results = self._simple_search(query_embedding, top_k)
            
            # Prefer results in requested language if available
            results = self._rerank_by_language(results, language)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def _simple_search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Fallback cosine similarity search"""
        from numpy.linalg import norm
        
        # Cosine similarity
        similarities = np.dot(self._embeddings, query_embedding) / (
            norm(self._embeddings, axis=1) * norm(query_embedding)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc['_score'] = float(similarities[idx])
            results.append(doc)
        
        return results
    
    def _rerank_by_language(self, results: List[Dict], language: str) -> List[Dict]:
        """Boost results that have content in the preferred language"""
        if language == 'en':
            return results
        
        # Check for language-specific fields
        lang_field = f"question_{language}"
        
        for doc in results:
            if lang_field in doc or f"answer_{language}" in doc:
                doc['_score'] = doc.get('_score', 0) * 1.2  # 20% boost
        
        return sorted(results, key=lambda x: x.get('_score', 0), reverse=True)
    
    def search_doctors(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search specifically for doctors"""
        # Filter to doctor documents
        doctor_docs = [d for d in self.documents if 'specialization' in d]
        
        if not doctor_docs:
            return []
        
        # Simple keyword matching as fallback
        query_lower = query.lower()
        results = []
        
        for doc in doctor_docs:
            score = 0
            searchable = f"{doc.get('name', '')} {doc.get('specialization', '')} {doc.get('department', '')}".lower()
            
            for word in query_lower.split():
                if word in searchable:
                    score += 1
            
            if score > 0:
                doc_copy = doc.copy()
                doc_copy['_score'] = score
                results.append(doc_copy)
        
        results.sort(key=lambda x: x['_score'], reverse=True)
        return results[:top_k]
    
    def format_results_for_prompt(self, results: List[Dict], doc_type: str = "faq") -> str:
        """Format search results for LLM prompt"""
        if not results:
            return ""
        
        formatted = []
        
        for i, doc in enumerate(results, 1):
            if doc_type == "faq" or 'question' in doc:
                q = doc.get('question', '')
                a = doc.get('answer', '')
                formatted.append(f"FAQ {i}:\nQ: {q}\nA: {a}")
            
            elif 'specialization' in doc:
                name = doc.get('name', 'Unknown')
                spec = doc.get('specialization', '')
                dept = doc.get('department', '')
                timing = doc.get('timings', '')
                days = doc.get('available_days', [])
                days_str = ', '.join(days) if isinstance(days, list) else days
                formatted.append(
                    f"Doctor {i}:\n"
                    f"Name: {name}\n"
                    f"Specialization: {spec}\n"
                    f"Department: {dept}\n"
                    f"Available: {days_str}\n"
                    f"Timing: {timing}"
                )
            
            elif 'name' in doc and 'floor' in doc:
                # Department
                name = doc.get('name', '')
                floor = doc.get('floor', '')
                timing = doc.get('timings', '')
                contact = doc.get('contact', '')
                formatted.append(
                    f"Department {i}:\n"
                    f"Name: {name}\n"
                    f"Location: {floor}\n"
                    f"Hours: {timing}\n"
                    f"Contact: {contact}"
                )
        
        return "\n\n".join(formatted)


# ============================================================================
# DOCUMENT LOADERS
# ============================================================================

def load_documents_from_redis(redis_store) -> List[Dict[str, Any]]:
    """Load all documents from Redis for RAG indexing"""
    documents = []
    
    # Load FAQs
    faqs = redis_store.get_faqs()
    for faq in faqs:
        faq['_type'] = 'faq'
        documents.append(faq)
    
    # Load doctors
    doctors = redis_store.get_doctors()
    for doc in doctors:
        doc['_type'] = 'doctor'
        documents.append(doc)
    
    # Load departments
    departments = redis_store.get_departments()
    for dept in departments:
        dept['_type'] = 'department'
        documents.append(dept)
    
    logger.info(f"Loaded {len(documents)} documents from Redis")
    return documents


def load_documents_from_json(filepath: str) -> List[Dict[str, Any]]:
    """Load documents from JSON file"""
    documents = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load FAQs
        for faq in data.get('faqs', []):
            faq['_type'] = 'faq'
            documents.append(faq)
        
        # Load doctors
        for doc in data.get('doctors', []):
            doc['_type'] = 'doctor'
            documents.append(doc)
        
        # Load departments
        for dept in data.get('departments', []):
            dept['_type'] = 'department'
            documents.append(dept)
        
        logger.info(f"Loaded {len(documents)} documents from {filepath}")
        
    except FileNotFoundError:
        logger.warning(f"Document file not found: {filepath}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error in {filepath}: {e}")
    
    return documents
