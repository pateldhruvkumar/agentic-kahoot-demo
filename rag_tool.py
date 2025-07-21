import chromadb
from typing import List, Dict, Tuple, Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import os
import re
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import numpy as np
from collections import Counter
import math

# Optional imports - graceful degradation if not available
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    print("⚠️ scikit-learn not available - some advanced features will be disabled")
    print("💡 Install with: pip install scikit-learn")
    HAS_SKLEARN = False
    TfidfVectorizer = None
    cosine_similarity = None

load_dotenv()

class RAGToolInput(BaseModel):
    query: str = Field(..., description="The query to search for relevant knowledge in the specified ChromaDB collection.")
    n_results: int = Field(12, description="Number of relevant documents to retrieve (balanced for speed and accuracy).")
    collection_name: str = Field(..., description="Name of the ChromaDB collection to query.")
    answer_choices: Optional[List[str]] = Field(None, description="Available answer choices to help with matching")
    smart_mode: bool = Field(True, description="Enable intelligent hybrid mode (fast + accuracy escalation)")
    min_confidence_threshold: float = Field(5.0, description="Minimum confidence for fast mode (escalates to full mode if below)")

class EnhancedRAGTool(BaseTool):
    name: str = "Enhanced ChromaDB RAG Tool"
    description: str = (
        "A highly advanced Retrieval-Augmented Generation (RAG) tool with semantic chunking, hybrid retrieval, "
        "intelligent query enhancement, and sophisticated answer matching. Uses multiple retrieval strategies "
        "including dense embeddings, sparse retrieval, and advanced scoring for maximum accuracy."
    )
    args_schema: Type[BaseModel] = RAGToolInput

    def __init__(self, collection_name: str, db_path: str = "chroma.sqlite3"):
        super().__init__()
        persist_dir = os.path.dirname(os.path.abspath(db_path)) or "."
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

        object.__setattr__(self, "_chroma_client", chromadb.PersistentClient(path=persist_dir))
        
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-ada-002"
        )
        object.__setattr__(self, "_embedding_function", openai_ef)
        object.__setattr__(self, "collection_name", collection_name)
        
        # Add simple cache for speed
        object.__setattr__(self, "_query_cache", {})
        object.__setattr__(self, "_cache_max_size", 50)
        
        # Initialize TF-IDF vectorizer for hybrid retrieval (if sklearn available)
        if HAS_SKLEARN:
            object.__setattr__(self, "_tfidf_vectorizer", TfidfVectorizer(
                max_features=5000,  # Reduced for speed
                stop_words='english',
                ngram_range=(1, 2),  # Reduced for speed
                max_df=0.8,
                min_df=1  # Reduced for speed
            ))
            object.__setattr__(self, "_tfidf_fitted", False)
            print("✅ Enhanced RAG with TF-IDF support initialized (speed optimized)")
        else:
            object.__setattr__(self, "_tfidf_vectorizer", None)
            object.__setattr__(self, "_tfidf_fitted", False)
            print("⚠️ Enhanced RAG initialized without TF-IDF (install scikit-learn for full features)")
        
        # Initialize spaCy for better text processing (optional)
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            object.__setattr__(self, "_nlp", nlp)
            object.__setattr__(self, "_has_spacy", True)
            print("✅ spaCy support enabled")
        except:
            object.__setattr__(self, "_nlp", None)
            object.__setattr__(self, "_has_spacy", False)
            print("⚠️ spaCy not available, using fallback text processing")

        try:
            object.__setattr__(self, "_collection", self._chroma_client.get_collection(
                name=collection_name,
                embedding_function=self._embedding_function
            ))
        except Exception as e:
            print(f"Warning: Could not get collection '{collection_name}': {e}")
            object.__setattr__(self, "_collection", None)

    def _enhanced_query_processing(self, query: str) -> Dict[str, List[str]]:
        """Enhanced query processing with better concept extraction and query reformulation"""
        clean_query = re.sub(r'\s+', ' ', query.strip())
        
        query_elements = {
            "original": [clean_query],
            "reformulated": [],
            "keywords": [],
            "phrases": [],
            "concepts": []
        }
        
        # Query reformulation patterns
        reformulations = self._generate_query_reformulations(clean_query)
        query_elements["reformulated"] = reformulations
        
        # Extract keywords and phrases
        keywords = self._extract_keywords(clean_query)
        phrases = self._extract_phrases(clean_query)
        
        query_elements["keywords"] = keywords
        query_elements["phrases"] = phrases
        
        # Extract concepts
        if self._has_spacy and self._nlp:
            concepts = self._extract_concepts_spacy(clean_query)
        else:
            concepts = self._extract_concepts_fallback(clean_query)
        
        query_elements["concepts"] = concepts
        
        return query_elements
    
    def _generate_query_reformulations(self, query: str) -> List[str]:
        """Generate alternative query formulations"""
        reformulations = []
        
        patterns = [
            (r"^What\s+(?:is|are|does|do|did|was|were)\s+", ""),
            (r"^How\s+(?:does|do|did|can|could)\s+", ""),
            (r"^Which\s+", ""),
            (r"^Where\s+(?:is|are|does|do)\s+", ""),
            (r"^When\s+(?:is|are|does|do|did)\s+", ""),
            (r"^Why\s+(?:is|are|does|do|did)\s+", ""),
            (r"\?\s*$", ""),
            (r"according\s+to\s+(?:the\s+)?(?:document|report|text)", ""),
            (r"as\s+mentioned\s+in\s+(?:the\s+)?(?:document|report|text)", ""),
        ]
        
        for pattern, replacement in patterns:
            reformulated = re.sub(pattern, replacement, query, flags=re.IGNORECASE).strip()
            if reformulated and reformulated != query and len(reformulated) > 3:
                reformulations.append(reformulated)
        
        # Add keyword-focused version
        keywords = self._extract_keywords(query)
        if keywords:
            keyword_query = " ".join(keywords[:5])
            reformulations.append(keyword_query)
        
        return reformulations[:3]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        stop_words = {
            'what', 'how', 'where', 'when', 'why', 'which', 'who', 'is', 'are', 'was', 'were',
            'does', 'do', 'did', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'says', 'said', 'according', 'mentioned', 'report',
            'document', 'states', 'indicates', 'that', 'this', 'these', 'those', 'can', 'could',
            'would', 'should', 'will', 'shall', 'may', 'might', 'must', 'have', 'has', 'had'
        }
        
        # Extract quoted terms first (highest priority)
        quoted_terms = re.findall(r'"([^"]*)"', text)
        keywords = [term.lower() for term in quoted_terms]
        
        # Extract regular words
        words = re.findall(r'\b\w{3,}\b', text.lower())
        keywords.extend([word for word in words if word not in stop_words])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords[:10]
    
    def _extract_phrases(self, text: str) -> List[str]:
        """Extract important phrases from text"""
        quoted_phrases = re.findall(r'"([^"]*)"', text)
        multi_word = re.findall(r'\b\w+\s+\w+(?:\s+\w+){0,2}\b', text)
        
        phrases = quoted_phrases + [phrase for phrase in multi_word if len(phrase.split()) >= 2]
        
        unique_phrases = []
        seen = set()
        for phrase in phrases:
            phrase_lower = phrase.lower().strip()
            if phrase_lower not in seen and len(phrase_lower) > 5:
                seen.add(phrase_lower)
                unique_phrases.append(phrase)
        
        return unique_phrases[:5]
    
    def _extract_concepts_spacy(self, text: str) -> List[str]:
        """Extract concepts using spaCy"""
        doc = self._nlp(text)
        
        concepts = []
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "PRODUCT", "WORK_OF_ART"]:
                concepts.append(ent.text)
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2 and len(chunk.text) > 5:
                concepts.append(chunk.text)
        
        return concepts[:5]
    
    def _extract_concepts_fallback(self, text: str) -> List[str]:
        """Fallback concept extraction"""
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        technical = re.findall(r'\b\w+[-_]\w+\b', text)
        numbers = re.findall(r'\b\d{4}\b|\b\d+\.\d+\b', text)
        
        concepts = capitalized + technical + numbers
        return concepts[:5]

    def _hybrid_retrieval(self, query_elements: Dict, n_results: int) -> Dict:
        """Implement hybrid retrieval combining dense embeddings with sparse retrieval"""
        all_results = []
        
        # Dense retrieval (semantic search)
        for query_type, queries in query_elements.items():
            for i, query in enumerate(queries):
                try:
                    semantic_results = self._collection.query(
                        query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
                    
                    if semantic_results.get("documents") and semantic_results["documents"][0]:
                        weight = self._get_query_weight(query_type, i)
                        for j, doc in enumerate(semantic_results["documents"][0]):
                            all_results.append({
                                "document": doc,
                                "metadata": semantic_results["metadatas"][0][j],
                                "distance": semantic_results["distances"][0][j],
                                "source": f"semantic-{query_type}-{i}",
                                "weight": weight
                            })
                except Exception as e:
                    print(f"   ❌ Semantic search failed for {query_type}: {e}")
        
        # Sparse retrieval (TF-IDF based)
        sparse_results = self._sparse_retrieval(query_elements, n_results)
        all_results.extend(sparse_results)
        
        # Merge and deduplicate results
        merged_results = self._merge_and_deduplicate_results(all_results, n_results)
        
        return merged_results
    
    def _fast_hybrid_retrieval(self, query_elements: Dict, n_results: int) -> Dict:
        """Optimized hybrid retrieval for speed"""
        all_results = []
        
        # Priority order: original query first (most likely to be relevant)
        priority_order = ["original", "keywords", "reformulated"]
        
        for query_type in priority_order:
            if query_type not in query_elements:
                continue
                
            queries = query_elements[query_type]
            # Limit to 1 query per type for speed
            for i, query in enumerate(queries[:1]):
                try:
                    semantic_results = self._collection.query(
                        query_texts=[query],
                        n_results=min(n_results, 5),  # Reduced for speed
                        include=['documents', 'metadatas', 'distances']
                    )
                    
                    if semantic_results.get("documents") and semantic_results["documents"][0]:
                        weight = self._get_query_weight(query_type, i)
                        for j, doc in enumerate(semantic_results["documents"][0]):
                            all_results.append({
                                "document": doc,
                                "metadata": semantic_results["metadatas"][0][j],
                                "distance": semantic_results["distances"][0][j],
                                "source": f"semantic-{query_type}",
                                "weight": weight
                            })
                except Exception as e:
                    print(f"   ❌ Fast semantic search failed for {query_type}: {e}")
        
        # Optional sparse retrieval (only if sklearn available and we have time)
        if HAS_SKLEARN and self._tfidf_vectorizer and len(all_results) < n_results:
            sparse_results = self._fast_sparse_retrieval(query_elements["original"][0], n_results)
            all_results.extend(sparse_results)
        
        # Quick merge and deduplicate
        merged_results = self._fast_merge_results(all_results, n_results)
        
        return merged_results
    
    def _fast_sparse_retrieval(self, query: str, n_results: int) -> List[Dict]:
        """Fast sparse retrieval with single query"""
        try:
            all_docs = self._collection.get()
            if not all_docs["documents"]:
                return []
            
            # Fit TF-IDF if not already done
            if not self._tfidf_fitted:
                self._tfidf_vectorizer.fit(all_docs["documents"])
                object.__setattr__(self, "_tfidf_fitted", True)
            
            doc_vectors = self._tfidf_vectorizer.transform(all_docs["documents"])
            query_vector = self._tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, doc_vectors)[0]
            
            # Get top 3 results quickly
            top_indices = np.argsort(similarities)[::-1][:3]
            
            sparse_results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    sparse_results.append({
                        "document": all_docs["documents"][idx],
                        "metadata": all_docs["metadatas"][idx],
                        "distance": 1.0 - similarities[idx],
                        "source": "sparse-fast",
                        "weight": 0.8
                    })
            
            return sparse_results
            
        except Exception as e:
            print(f"   ❌ Fast sparse retrieval failed: {e}")
            return []
    
    def _fast_merge_results(self, all_results: List[Dict], n_results: int) -> Dict:
        """Quick result merging without complex deduplication"""
        # Simple sort by weighted score
        for result in all_results:
            result["weighted_score"] = result["distance"] / result["weight"]
        
        # Sort and take top results
        sorted_results = sorted(all_results, key=lambda x: x["weighted_score"])[:n_results]
        
        documents = [r["document"] for r in sorted_results]
        metadatas = [r["metadata"] for r in sorted_results]
        distances = [r["weighted_score"] for r in sorted_results]
        
        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances]
        }
    
    def _get_query_weight(self, query_type: str, index: int) -> float:
        """Assign weights to different query types"""
        weights = {
            "original": 1.0,
            "reformulated": 0.9 - (index * 0.1),
            "phrases": 0.95,
            "concepts": 0.8,
            "keywords": 0.7
        }
        return weights.get(query_type, 0.5)
    
    def _sparse_retrieval(self, query_elements: Dict, n_results: int) -> List[Dict]:
        """Implement sparse retrieval using TF-IDF (if available)"""
        if not HAS_SKLEARN or not self._tfidf_vectorizer:
            print("   ⚠️ TF-IDF sparse retrieval not available (scikit-learn not installed)")
            return []
            
        try:
            all_docs = self._collection.get()
            if not all_docs["documents"]:
                return []
            
            # Fit TF-IDF if not already done
            if not self._tfidf_fitted:
                self._tfidf_vectorizer.fit(all_docs["documents"])
                object.__setattr__(self, "_tfidf_fitted", True)
            
            doc_vectors = self._tfidf_vectorizer.transform(all_docs["documents"])
            sparse_results = []
            
            # Search with different query formulations
            for query_type, queries in query_elements.items():
                for i, query in enumerate(queries):
                    try:
                        query_vector = self._tfidf_vectorizer.transform([query])
                        similarities = cosine_similarity(query_vector, doc_vectors)[0]
                        
                        top_indices = np.argsort(similarities)[::-1][:n_results]
                        weight = self._get_query_weight(query_type, i) * 0.8
                        
                        for idx in top_indices:
                            if similarities[idx] > 0.1:
                                sparse_results.append({
                                    "document": all_docs["documents"][idx],
                                    "metadata": all_docs["metadatas"][idx],
                                    "distance": 1.0 - similarities[idx],
                                    "source": f"sparse-{query_type}-{i}",
                                    "weight": weight
                                })
                    except Exception as e:
                        print(f"   ❌ Sparse retrieval failed for {query}: {e}")
            
            return sparse_results
            
        except Exception as e:
            print(f"   ❌ Sparse retrieval system failed: {e}")
            return []
    
    def _merge_and_deduplicate_results(self, all_results: List[Dict], n_results: int) -> Dict:
        """Merge and deduplicate results from different retrieval methods"""
        unique_results = {}
        
        for result in all_results:
            doc_key = result["document"][:200]
            weighted_score = result["distance"] / result["weight"]
            
            if doc_key not in unique_results or weighted_score < unique_results[doc_key]["weighted_score"]:
                result["weighted_score"] = weighted_score
                unique_results[doc_key] = result
        
        sorted_results = sorted(unique_results.values(), key=lambda x: x["weighted_score"])[:n_results]
        
        documents = [r["document"] for r in sorted_results]
        metadatas = [r["metadata"] for r in sorted_results]
        distances = [r["weighted_score"] for r in sorted_results]
        
        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances]
        }

    def _advanced_answer_choice_analysis(self, query: str, answer_choices: List[str], search_results: Dict) -> Dict:
        """Advanced answer choice analysis with contextual similarity and evidence aggregation"""
        if not answer_choices or not search_results.get("documents") or not search_results["documents"][0]:
            return {}
        
        choice_analysis = {}
        
        for choice in answer_choices:
            scores = {
                "exact_matches": 0,
                "partial_matches": 0,
                "context_similarity": 0,
                "evidence_count": 0,
                "confidence": 0
            }
            
            choice_lower = choice.lower()
            choice_words = set(re.findall(r'\b\w{3,}\b', choice_lower))
            
            # Analyze each document
            for i, doc in enumerate(search_results["documents"][0][:10]):
                doc_lower = doc.lower()
                doc_words = set(re.findall(r'\b\w{3,}\b', doc_lower))
                
                distance = search_results["distances"][0][i] if search_results.get("distances") else 1.0
                doc_weight = 1.0 / (1.0 + distance)
                
                # Exact phrase matching
                if choice_lower in doc_lower:
                    scores["exact_matches"] += doc_weight * 10
                    scores["evidence_count"] += 1
                
                # Word overlap analysis
                if choice_words and doc_words:
                    overlap = choice_words.intersection(doc_words)
                    if overlap:
                        overlap_ratio = len(overlap) / len(choice_words)
                        scores["partial_matches"] += overlap_ratio * doc_weight * 5
                        
                        if overlap_ratio > 0.7:
                            scores["evidence_count"] += 1
                
                # Contextual similarity using TF-IDF
                if HAS_SKLEARN and self._tfidf_fitted:
                    try:
                        choice_vector = self._tfidf_vectorizer.transform([choice])
                        doc_vector = self._tfidf_vectorizer.transform([doc])
                        similarity = cosine_similarity(choice_vector, doc_vector)[0][0]
                        scores["context_similarity"] += similarity * doc_weight * 3
                    except Exception:
                        pass
                
                # Proximity bonus
                query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
                if query_words and choice_words:
                    sentences = re.split(r'[.!?]', doc_lower)
                    for sentence in sentences:
                        sentence_words = set(re.findall(r'\b\w{3,}\b', sentence))
                        if (query_words.intersection(sentence_words) and 
                            choice_words.intersection(sentence_words)):
                            scores["context_similarity"] += doc_weight * 2
            
            # Calculate overall confidence score
            scores["confidence"] = (
                scores["exact_matches"] * 0.4 +
                scores["partial_matches"] * 0.3 +
                scores["context_similarity"] * 0.2 +
                scores["evidence_count"] * 0.1
            )
            
            choice_analysis[choice] = scores
        
        return choice_analysis

    def _context_synthesis(self, search_results: Dict, query: str, max_context_length: int = 2000) -> str:
        """Intelligently synthesize context from multiple retrieved chunks"""
        if not search_results.get("documents") or not search_results["documents"][0]:
            return ""
        
        relevant_contexts = []
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        
        for i, doc in enumerate(search_results["documents"][0][:5]):
            sentences = re.split(r'[.!?]+', doc)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                
                sentence_words = set(re.findall(r'\b\w{3,}\b', sentence.lower()))
                overlap = query_words.intersection(sentence_words)
                
                if overlap:
                    relevance_score = len(overlap) / len(query_words) if query_words else 0
                    distance_weight = 1.0 / (1.0 + search_results["distances"][0][i])
                    final_score = relevance_score * distance_weight
                    
                    relevant_contexts.append({
                        "sentence": sentence,
                        "score": final_score,
                        "doc_index": i
                    })
        
        relevant_contexts.sort(key=lambda x: x["score"], reverse=True)
        
        synthesized_context = ""
        used_docs = set()
        
        for context in relevant_contexts:
            if len(synthesized_context) + len(context["sentence"]) > max_context_length:
                break
            
            if context["doc_index"] not in used_docs or len(used_docs) < 3:
                synthesized_context += context["sentence"] + ". "
                used_docs.add(context["doc_index"])
        
        return synthesized_context.strip()

    def _fast_answer_choice_analysis(self, query: str, answer_choices: List[str], search_results: Dict) -> Dict:
        """Fast answer choice analysis focused on exact matches"""
        if not answer_choices or not search_results.get("documents") or not search_results["documents"][0]:
            return {}
        
        choice_analysis = {}
        
        for choice in answer_choices:
            score = 0
            choice_lower = choice.lower()
            
            # Focus on exact matches in top 3 documents for speed
            for i, doc in enumerate(search_results["documents"][0][:3]):
                doc_lower = doc.lower()
                
                # Distance-based weighting
                distance = search_results["distances"][0][i] if search_results.get("distances") else 1.0
                doc_weight = 1.0 / (1.0 + distance)
                
                # Exact phrase matching (highest priority)
                if choice_lower in doc_lower:
                    score += doc_weight * 10
                
                # Quick word overlap check
                choice_words = set(choice_lower.split())
                if choice_words and any(word in doc_lower for word in choice_words if len(word) > 2):
                    score += doc_weight * 3
            
            choice_analysis[choice] = {"confidence": score, "evidence_count": 1 if score > 5 else 0}
        
        return choice_analysis

    def _intelligent_hybrid_processing(self, query: str, answer_choices: Optional[List[str]], n_results: int, min_confidence: float) -> Dict:
        """
        Intelligent hybrid processing: Start fast, escalate to full mode if confidence is low
        This ensures both speed AND accuracy
        """
        print(f"🧠 INTELLIGENT HYBRID MODE:")
        
        # PHASE 1: Fast processing attempt
        print(f"   🚀 Phase 1: Fast processing...")
        fast_query_elements = self._fast_query_processing(query)
        fast_results = self._fast_hybrid_retrieval(fast_query_elements, min(n_results, 8))
        
        # Quick confidence check
        if not fast_results.get("documents") or not fast_results["documents"][0]:
            print(f"   ❌ Fast mode failed - no results. Escalating to full mode...")
            return self._full_accuracy_processing(query, answer_choices, n_results)
        
        # Analyze answer choices quickly
        fast_choice_analysis = {}
        if answer_choices:
            fast_choice_analysis = self._fast_answer_choice_analysis(query, answer_choices, fast_results)
            
            if fast_choice_analysis:
                best_confidence = max(choice['confidence'] for choice in fast_choice_analysis.values())
                print(f"   📊 Fast mode confidence: {best_confidence:.1f} (threshold: {min_confidence})")
                
                # If confidence is high enough, use fast results
                if best_confidence >= min_confidence:
                    print(f"   ✅ High confidence achieved! Using fast results.")
                    return {
                        "results": fast_results,
                        "choice_analysis": fast_choice_analysis,
                        "processing_mode": "fast",
                        "confidence": best_confidence
                    }
        
        # PHASE 2: Full accuracy processing
        print(f"   🎯 Phase 2: Full accuracy processing (low confidence detected)...")
        return self._full_accuracy_processing(query, answer_choices, n_results)
    
    def _full_accuracy_processing(self, query: str, answer_choices: Optional[List[str]], n_results: int) -> Dict:
        """
        Full accuracy processing with comprehensive analysis
        """
        # Enhanced query processing with more reformulations
        enhanced_query_elements = self._enhanced_query_processing(query)
        
        # Add additional query variations for accuracy
        enhanced_query_elements["accuracy_variants"] = self._generate_accuracy_variants(query)
        
        # Full hybrid retrieval with larger scope
        full_results = self._hybrid_retrieval(enhanced_query_elements, n_results)
        
        # Advanced answer choice analysis
        full_choice_analysis = {}
        if answer_choices:
            full_choice_analysis = self._advanced_answer_choice_analysis(query, answer_choices, full_results)
            
            # Add cross-validation for accuracy
            full_choice_analysis = self._cross_validate_answers(query, answer_choices, full_results, full_choice_analysis)
        
        best_confidence = 0
        if full_choice_analysis:
            best_confidence = max(choice['confidence'] for choice in full_choice_analysis.values())
        
        print(f"   📈 Full mode confidence: {best_confidence:.1f}")
        
        return {
            "results": full_results,
            "choice_analysis": full_choice_analysis,
            "processing_mode": "full",
            "confidence": best_confidence
        }
    
    def _generate_accuracy_variants(self, query: str) -> List[str]:
        """Generate additional query variants focused on accuracy"""
        variants = []
        
        # Add more comprehensive reformulations
        base_patterns = [
            (r"^What\s+(?:is|are|does|do|did|was|were|can|could|will|would|should|might|may)\s+", ""),
            (r"^How\s+(?:does|do|did|can|could|will|would|should|might|may)\s+", ""),
            (r"^Which\s+(?:is|are|does|do|did|was|were|can|could|will|would)\s+", ""),
            (r"^Where\s+(?:is|are|does|do|did|was|were|can|could)\s+", ""),
            (r"^When\s+(?:is|are|does|do|did|was|were|will|would)\s+", ""),
            (r"^Why\s+(?:is|are|does|do|did|was|were|can|could|will|would)\s+", ""),
        ]
        
        for pattern, replacement in base_patterns:
            variant = re.sub(pattern, replacement, query, flags=re.IGNORECASE).strip()
            if variant and variant != query and len(variant) > 3:
                variants.append(variant)
        
        # Add context-aware variants
        if "according to" in query.lower():
            clean_variant = re.sub(r"according\s+to\s+(?:the\s+)?(?:document|report|text|source)", "", query, flags=re.IGNORECASE).strip()
            if clean_variant:
                variants.append(clean_variant)
        
        # Add key concept combinations
        key_concepts = self._extract_keywords(query)
        if len(key_concepts) >= 2:
            # Create combinations of key concepts
            variants.append(" ".join(key_concepts[:3]))
            variants.append(" ".join(key_concepts[-3:]))
        
        return variants[:5]  # Limit to top 5 for efficiency
    
    def _cross_validate_answers(self, query: str, answer_choices: List[str], search_results: Dict, choice_analysis: Dict) -> Dict:
        """
        Cross-validate answer choices using multiple validation methods
        """
        for choice in answer_choices:
            if choice in choice_analysis:
                original_confidence = choice_analysis[choice]['confidence']
                
                # Validation 1: Context window analysis
                context_score = self._validate_in_context_window(query, choice, search_results)
                
                # Validation 2: Semantic consistency check
                semantic_score = self._validate_semantic_consistency(query, choice, search_results)
                
                # Validation 3: Evidence strength
                evidence_score = self._validate_evidence_strength(choice, search_results)
                
                # Combine scores with weighted average
                validated_confidence = (
                    original_confidence * 0.5 +
                    context_score * 0.2 +
                    semantic_score * 0.2 +
                    evidence_score * 0.1
                )
                
                choice_analysis[choice]['confidence'] = validated_confidence
                choice_analysis[choice]['validation_scores'] = {
                    'original': original_confidence,
                    'context': context_score,
                    'semantic': semantic_score,
                    'evidence': evidence_score
                }
        
        return choice_analysis
    
    def _validate_in_context_window(self, query: str, choice: str, search_results: Dict) -> float:
        """Validate choice within context windows of documents"""
        score = 0
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        choice_words = set(re.findall(r'\b\w{3,}\b', choice.lower()))
        
        if not search_results.get("documents") or not search_results["documents"][0]:
            return score
        
        for doc in search_results["documents"][0][:5]:
            # Split into sentences and check context windows
            sentences = re.split(r'[.!?]+', doc.lower())
            for sentence in sentences:
                sentence_words = set(re.findall(r'\b\w{3,}\b', sentence))
                
                # Check if both query and choice concepts appear in same sentence
                query_overlap = query_words.intersection(sentence_words)
                choice_overlap = choice_words.intersection(sentence_words)
                
                if query_overlap and choice_overlap:
                    score += len(query_overlap) + len(choice_overlap)
        
        return min(score, 10)  # Cap at 10
    
    def _validate_semantic_consistency(self, query: str, choice: str, search_results: Dict) -> float:
        """Validate semantic consistency between query, choice, and documents"""
        if not HAS_SKLEARN or not self._tfidf_fitted:
            return 5.0  # Default neutral score if TF-IDF unavailable
        
        try:
            # Create combined text from top documents
            combined_docs = " ".join(search_results["documents"][0][:3])
            
            # Calculate semantic similarity
            texts = [query, choice, combined_docs]
            vectors = self._tfidf_vectorizer.transform(texts)
            
            # Query-choice similarity
            qc_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            # Choice-documents similarity
            cd_sim = cosine_similarity(vectors[1:2], vectors[2:3])[0][0]
            
            # Combined semantic score
            semantic_score = (qc_sim + cd_sim) * 5  # Scale to 0-10 range
            return min(semantic_score, 10)
            
        except Exception:
            return 5.0  # Default if calculation fails
    
    def _validate_evidence_strength(self, choice: str, search_results: Dict) -> float:
        """Validate the strength of evidence for the choice"""
        if not search_results.get("documents") or not search_results["documents"][0]:
            return 0
        
        choice_lower = choice.lower()
        evidence_count = 0
        evidence_quality = 0
        
        for i, doc in enumerate(search_results["documents"][0][:5]):
            doc_lower = doc.lower()
            
            # Check for exact phrase
            if choice_lower in doc_lower:
                evidence_count += 2  # Strong evidence
                evidence_quality += (5 - i)  # Earlier documents weighted more
            
            # Check for word components
            choice_words = choice_lower.split()
            word_matches = sum(1 for word in choice_words if len(word) > 2 and word in doc_lower)
            if word_matches >= len(choice_words) * 0.7:  # 70% word match
                evidence_count += 1
                evidence_quality += (3 - i) * 0.5
        
        return min(evidence_count + evidence_quality, 10)

    def _run(self, query: str, n_results: int = 12, collection_name: str = None, answer_choices: Optional[List[str]] = None, smart_mode: bool = True, min_confidence_threshold: float = 5.0) -> List[str]:
        """Intelligent hybrid processing: Fast response with accuracy guarantee"""
        collection_to_query = self._collection

        if not collection_to_query:
            return ["❌ Error: No collection selected for query."]

        # Check cache first (always fast)
        cache_key = self._get_cache_key(query, answer_choices)
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result:
            print(f"🚀 CACHE HIT - Returning cached result for: '{query[:50]}...'")
            return cached_result["response"]

        print(f"\n🧠 INTELLIGENT HYBRID RAG PROCESSING:")
        print(f"   Query: '{query}'")
        print(f"   Smart mode: {smart_mode}")
        print(f"   Confidence threshold: {min_confidence_threshold}")
        print(f"   Answer choices: {len(answer_choices) if answer_choices else 0}")
        
        # Use intelligent hybrid processing
        if smart_mode and answer_choices:
            processing_result = self._intelligent_hybrid_processing(query, answer_choices, n_results, min_confidence_threshold)
        else:
            # Fallback to full processing if no answer choices or smart mode disabled
            processing_result = self._full_accuracy_processing(query, answer_choices, n_results)
        
        final_results = processing_result["results"]
        choice_analysis = processing_result["choice_analysis"]
        processing_mode = processing_result["processing_mode"]
        final_confidence = processing_result["confidence"]
        
        print(f"   🎯 Final mode used: {processing_mode}")
        print(f"   📊 Final confidence: {final_confidence:.1f}")
        
        if not final_results.get("documents") or not final_results["documents"][0]:
            return ["❌ No relevant documents found after comprehensive analysis."]
        
        # Build comprehensive response
        docs_with_metadata = []
        
        # Header with processing info
        mode_emoji = "🚀" if processing_mode == "fast" else "🎯" if processing_mode == "full" else "🧠"
        docs_with_metadata.append(f"{mode_emoji} {processing_mode.upper()} MODE: Found {len(final_results['documents'][0])} documents")
        
        # Add confidence and accuracy indicators
        if final_confidence >= 8.0:
            confidence_indicator = "🎯 VERY HIGH CONFIDENCE"
        elif final_confidence >= 5.0:
            confidence_indicator = "📈 HIGH CONFIDENCE" 
        elif final_confidence >= 3.0:
            confidence_indicator = "📊 MEDIUM CONFIDENCE"
        else:
            confidence_indicator = "⚠️ LOW CONFIDENCE"
        
        docs_with_metadata.append(f"{confidence_indicator} (Score: {final_confidence:.1f})")
        
        # Add top recommendation with validation info
        if choice_analysis:
            sorted_choices = sorted(choice_analysis.items(), key=lambda x: x[1]['confidence'], reverse=True)
            best_choice = sorted_choices[0]
            best_score = best_choice[1]['confidence']
            
            # Determine recommendation strength
            if best_score >= 8.0:
                recommendation_text = f"🎯 HIGHLY RECOMMENDED ANSWER: '{best_choice[0]}'"
            elif best_score >= 5.0:
                recommendation_text = f"📈 RECOMMENDED ANSWER: '{best_choice[0]}'"
            elif best_score >= 3.0:
                recommendation_text = f"📊 BEST OPTION: '{best_choice[0]}'"
            else:
                recommendation_text = f"⚠️ UNCERTAIN GUESS: '{best_choice[0]}'"
            
            docs_with_metadata.append(f"{recommendation_text} (confidence: {best_score:.1f})")
            
            # Add validation details if available
            if 'validation_scores' in best_choice[1]:
                val_scores = best_choice[1]['validation_scores']
                docs_with_metadata.append(f"   🔍 Validation: Context({val_scores['context']:.1f}) Semantic({val_scores['semantic']:.1f}) Evidence({val_scores['evidence']:.1f})")
            
            # Show all choices with confidence scores
            docs_with_metadata.append("📊 All answer choices:")
            for choice, scores in sorted_choices:
                conf = scores['confidence']
                if conf >= 8.0:
                    emoji = "🎯"
                elif conf >= 5.0:
                    emoji = "📈"
                elif conf >= 3.0:
                    emoji = "📊"
                else:
                    emoji = "📌"
                docs_with_metadata.append(f"   {emoji} {choice}: {conf:.1f}")
        
        docs_with_metadata.append("="*60)
        
        # Show evidence with quality indicators
        evidence_limit = 5 if final_confidence >= 5.0 else 7  # More evidence for lower confidence
        
        for i in range(min(evidence_limit, len(final_results["documents"][0]))):
            doc_content = final_results["documents"][0][i]
            distance = final_results["distances"][0][i] if final_results.get("distances") else "N/A"
            
            # Quality indicator
            if distance < 0.3:
                quality = "🎯 EXCELLENT"
            elif distance < 0.6:
                quality = "📈 GOOD"
            else:
                quality = "📊 FAIR"
            
            # Highlight answer choice matches with confidence indicators
            content_preview = doc_content[:300]  # Longer preview for accuracy
            if answer_choices:
                for choice in answer_choices:
                    if choice.lower() in content_preview.lower():
                        # Use different highlighting based on choice confidence
                        if choice in choice_analysis:
                            choice_conf = choice_analysis[choice]['confidence']
                            if choice_conf >= 8.0:
                                highlight = f"🎯🎯🎯 {choice} 🎯🎯🎯"
                            elif choice_conf >= 5.0:
                                highlight = f"📈📈 {choice} 📈📈"
                            else:
                                highlight = f"📊 {choice} 📊"
                            content_preview = content_preview.replace(choice, highlight)
            
            docs_with_metadata.append(f"📄 Evidence {i+1} ({quality}): {content_preview}... (score: {distance:.3f})")
        
        # Add processing summary
        docs_with_metadata.append("="*60)
        docs_with_metadata.append(f"🔬 PROCESSING SUMMARY:")
        docs_with_metadata.append(f"   Mode: {processing_mode} processing")
        docs_with_metadata.append(f"   Confidence: {final_confidence:.1f}/10.0")
        docs_with_metadata.append(f"   Evidence quality: {evidence_limit} documents analyzed")
        
        # Cache the high-quality result
        cache_result = {
            "response": docs_with_metadata,
            "choice_analysis": choice_analysis,
            "confidence": final_confidence,
            "processing_mode": processing_mode
        }
        self._add_to_cache(cache_key, cache_result)
        
        return docs_with_metadata

    def _calculate_confidence_score(self, results: Dict, query: str) -> float:
        """Calculate confidence score based on result quality"""
        if not results.get("documents") or not results["documents"][0]:
            return 0.0
        
        num_results = len(results["documents"][0])
        avg_distance = sum(results["distances"][0]) / num_results if results.get("distances") and results["distances"][0] else 1.0
        
        query_lower = query.lower()
        exact_matches = 0
        for doc in results["documents"][0][:3]:
            if any(phrase.lower() in doc.lower() for phrase in re.findall(r'\b\w+(?:\s+\w+){1,3}\b', query)):
                exact_matches += 1
        
        distance_score = max(0, 1.0 - avg_distance)
        match_score = exact_matches / min(3, num_results)
        result_count_score = min(1.0, num_results / 5.0)
        
        confidence = (distance_score * 0.5) + (match_score * 0.3) + (result_count_score * 0.2)
        return min(1.0, confidence)

    def _get_cache_key(self, query: str, answer_choices: Optional[List[str]]) -> str:
        """Generate cache key for query and answer choices"""
        choices_str = "|".join(sorted(answer_choices)) if answer_choices else ""
        return f"{query.lower().strip()}:{choices_str}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Get result from cache if available"""
        return self._query_cache.get(cache_key)
    
    def _add_to_cache(self, cache_key: str, result: Dict):
        """Add result to cache with size management"""
        if len(self._query_cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
        
        self._query_cache[cache_key] = result

    def _fast_query_processing(self, query: str) -> Dict[str, List[str]]:
        """Streamlined query processing for speed"""
        clean_query = re.sub(r'\s+', ' ', query.strip())
        
        # Fast mode: only generate 1-2 key reformulations
        query_elements = {
            "original": [clean_query],
            "keywords": [],
            "reformulated": []
        }
        
        # Quick keyword extraction
        keywords = self._extract_keywords(clean_query)
        if keywords:
            query_elements["keywords"] = keywords[:3]  # Only top 3 for speed
        
        # Single reformulation
        reformulated = self._quick_reformulate(clean_query)
        if reformulated:
            query_elements["reformulated"] = [reformulated]
        
        return query_elements
    
    def _quick_reformulate(self, query: str) -> Optional[str]:
        """Single fast reformulation"""
        # Remove question words quickly
        reformulated = re.sub(r"^(What|How|Which|Where|When|Why)\s+(?:is|are|does|do|did|was|were)?\s*", "", query, flags=re.IGNORECASE)
        reformulated = re.sub(r"\?\s*$", "", reformulated).strip()
        
        if reformulated and reformulated != query and len(reformulated) > 3:
            return reformulated
        return None

# Backward compatibility - create alias for the old class name
RAGTool = EnhancedRAGTool
