import chromadb
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import os
import re
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

class RAGToolInput(BaseModel):
    query: str = Field(..., description="The query to search for relevant knowledge in the specified ChromaDB collection.")
    n_results: int = Field(15, description="Number of relevant documents to retrieve.")
    collection_name: str = Field(..., description="Name of the ChromaDB collection to query.")
    answer_choices: Optional[List[str]] = Field(None, description="Available answer choices to help with matching")

class RAGTool(BaseTool):
    name: str = "ChromaDB RAG Tool"
    description: str = (
        "An advanced Retrieval-Augmented Generation (RAG) tool for CrewAI with multiple search strategies and answer matching. "
        "Queries a specified ChromaDB SQLite collection for relevant knowledge using semantic, keyword, phrase, and exact text matching. "
        "Use the 'query' argument to search for information, 'collection_name' to specify the collection, and optionally 'answer_choices' for better matching."
    )
    args_schema = RAGToolInput

    def __init__(self, collection_name: str, db_path: str = "chroma.sqlite3"):
        super().__init__()
        persist_dir = os.path.dirname(os.path.abspath(db_path)) or "."
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

        # Ensure client and embedding_function are attached to the instance using object.__setattr__
        # to avoid potential issues with BaseTool's __setattr__ or Pydantic interference.
        object.__setattr__(self, "_chroma_client", chromadb.PersistentClient(path=persist_dir))
        
        # Use ChromaDB's built-in OpenAI embedding function
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-ada-002"
        )
        object.__setattr__(self, "_embedding_function", openai_ef)

        # Initialize collection_name and collection attribute
        object.__setattr__(self, "collection_name", collection_name)
        try:
            object.__setattr__(self, "_collection", self._chroma_client.get_collection(
                name=collection_name,
                embedding_function=self._embedding_function
            ))
        except Exception as e:
            print(f"Warning: Could not get collection '{collection_name}': {e}")
            object.__setattr__(self, "_collection", None)

    def _expand_query(self, query: str) -> List[str]:
        """Generate alternative query phrasings to catch different ways information might be presented"""
        base_query = query.strip()
        expanded_queries = [base_query]
        
        # Generate variations by removing question words and rephrasing
        question_patterns = [
            (r"^What\s+(?:does\s+)?(?:the\s+)?(?:report\s+)?(?:say\s+)?(?:is\s+)?", ""),
            (r"^What\s+is\s+", ""),
            (r"^How\s+(?:does\s+)?", ""),
            (r"^Which\s+", ""),
            (r"^Where\s+", ""),
            (r"according\s+to\s+the\s+(?:report|document)", ""),
            (r"as\s+mentioned\s+(?:in|within)\s+(?:the\s+)?(?:document|report)", ""),
            (r"\?\s*$", ""),
        ]
        
        for pattern, replacement in question_patterns:
            alternative = re.sub(pattern, replacement, base_query, flags=re.IGNORECASE).strip()
            if alternative and alternative != base_query and len(alternative) > 5:
                expanded_queries.append(alternative)
        
        # Add key concept extraction
        key_concepts = self._extract_key_concepts(base_query)
        if key_concepts:
            expanded_queries.append(" ".join(key_concepts))
        
        return expanded_queries

    def _extract_key_concepts(self, query: str) -> List[str]:
        """Extract the most important concepts from the query"""
        # Remove question words and common phrases
        stop_phrases = [
            'what is', 'what does', 'how does', 'where is', 'when is', 'why is',
            'according to', 'mentioned in', 'the report says', 'the document',
            'as mentioned', 'within the document'
        ]
        
        cleaned_query = query.lower()
        for phrase in stop_phrases:
            cleaned_query = cleaned_query.replace(phrase, ' ')
        
        # Extract important terms (3+ chars, not common words)
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'has', 'let', 'put', 'say', 'she', 'too', 'use'}
        
        words = re.findall(r'\b\w{3,}\b', cleaned_query)
        concepts = [word for word in words if word not in stop_words]
        
        return concepts[:5]  # Return top 5 concepts

    def _exact_text_search(self, query: str, n_results: int) -> Dict:
        """Search for documents containing exact phrases from the query"""
        # Extract important phrases (2+ words) for exact matching
        phrases = []
        
        # Get quoted phrases first
        quoted = re.findall(r'"([^"]*)"', query)
        phrases.extend([q for q in quoted if len(q.split()) >= 2])
        
        # Get important multi-word phrases
        multi_word = re.findall(r'\b\w+\s+\w+(?:\s+\w+)*\b', query)
        long_phrases = [p for p in multi_word if len(p.split()) >= 2 and len(p) > 8]
        phrases.extend(long_phrases[:3])  # Top 3 phrases
        
        if not phrases:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        # Use the longest phrase for exact search
        search_phrase = max(phrases, key=len)
        
        try:
            # Use the phrase as a direct query to find exact matches
            results = self._collection.query(
                query_texts=[search_phrase],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            return results
        except Exception as e:
            print(f"   ❌ Exact text search failed: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    def _analyze_answer_choices(self, query: str, answer_choices: List[str], search_results: Dict) -> Dict:
        """Analyze how well each answer choice matches the search results"""
        if not answer_choices or not search_results.get("documents") or not search_results["documents"][0]:
            return {}
        
        choice_scores = {}
        
        for choice in answer_choices:
            score = 0
            choice_lower = choice.lower()
            
            # Check direct matches in top documents
            for i, doc in enumerate(search_results["documents"][0][:5]):  # Check top 5 docs
                doc_lower = doc.lower()
                
                # Exact phrase match (highest score)
                if choice_lower in doc_lower:
                    score += 10 / (i + 1)  # Higher score for earlier results
                
                # Partial word matches
                choice_words = set(re.findall(r'\b\w{3,}\b', choice_lower))
                doc_words = set(re.findall(r'\b\w{3,}\b', doc_lower))
                common_words = choice_words.intersection(doc_words)
                
                if common_words:
                    word_match_score = len(common_words) / len(choice_words) if choice_words else 0
                    score += word_match_score * 3 / (i + 1)
            
            choice_scores[choice] = score
        
        return choice_scores

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms and phrases from the query for better search"""
        # Remove common question words
        stop_words = {'what', 'how', 'where', 'when', 'why', 'which', 'who', 'is', 'are', 'was', 'were', 'does', 'do', 'did', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'says', 'said', 'according', 'mentioned', 'report', 'document', 'states', 'indicates'}
        
        # Extract phrases in quotes first (highest priority)
        quoted_phrases = re.findall(r'"([^"]*)"', query)
        
        # Extract numbered items (e.g., "2025", "Level 1", etc.)
        numbers = re.findall(r'\b\d{4}\b|\b\d+\b', query)
        
        # Extract capitalized terms (likely proper nouns/important concepts)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        
        # Extract technical terms and compound words
        technical_terms = re.findall(r'\b\w+[-_]\w+\b', query)
        
        # Extract important terms (longer than 2 characters, not stop words)
        words = re.findall(r'\b\w{3,}\b', query.lower())
        key_terms = [word for word in words if word not in stop_words]
        
        # Combine all extracted terms with priority order
        search_terms = quoted_phrases + numbers + capitalized + technical_terms + key_terms
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in search_terms:
            term_lower = term.lower()
            if term_lower not in seen:
                seen.add(term_lower)
                unique_terms.append(term)
        
        return unique_terms[:10]  # Limit to top 10 terms

    def _calculate_confidence_score(self, results: Dict, query: str) -> float:
        """Calculate confidence score based on result quality"""
        if not results.get("documents") or not results["documents"][0]:
            return 0.0
        
        # Factors for confidence calculation
        num_results = len(results["documents"][0])
        avg_distance = sum(results["distances"][0]) / num_results if results.get("distances") and results["distances"][0] else 1.0
        
        # Check for exact phrase matches in top results
        query_lower = query.lower()
        exact_matches = 0
        for doc in results["documents"][0][:3]:  # Check top 3 results
            if any(phrase.lower() in doc.lower() for phrase in re.findall(r'\b\w+(?:\s+\w+){1,3}\b', query)):
                exact_matches += 1
        
        # Calculate base confidence
        distance_score = max(0, 1.0 - avg_distance)  # Lower distance = higher confidence
        match_score = exact_matches / min(3, num_results)  # Proportion of exact matches
        result_count_score = min(1.0, num_results / 5.0)  # More results = higher confidence, cap at 5
        
        # Weighted combination
        confidence = (distance_score * 0.5) + (match_score * 0.3) + (result_count_score * 0.2)
        return min(1.0, confidence)

    def _semantic_search(self, query: str, n_results: int) -> Dict:
        """Perform semantic search using embeddings"""
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            return results
        except Exception as e:
            print(f"   ❌ Semantic search failed: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    def _keyword_search(self, query: str, n_results: int) -> Dict:
        """Perform keyword-based search using key terms"""
        key_terms = self._extract_key_terms(query)
        
        if not key_terms:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        # Create keyword query by joining terms
        keyword_query = " ".join(key_terms[:5])  # Use top 5 key terms
        
        try:
            results = self._collection.query(
                query_texts=[keyword_query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            return results
        except Exception as e:
            print(f"   ❌ Keyword search failed: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    def _phrase_search(self, query: str, n_results: int) -> Dict:
        """Search for exact phrases or important terms"""
        # Extract important phrases (2+ words)
        phrases = re.findall(r'\b\w+\s+\w+(?:\s+\w+)*\b', query)
        long_phrases = [phrase for phrase in phrases if len(phrase.split()) >= 2]
        
        if not long_phrases:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        # Use the longest phrase for search
        phrase_query = max(long_phrases, key=len)
        
        try:
            results = self._collection.query(
                query_texts=[phrase_query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            return results
        except Exception as e:
            print(f"   ❌ Phrase search failed: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    def _merge_and_rank_results(self, semantic_results: Dict, keyword_results: Dict, phrase_results: Dict, n_results: int) -> Dict:
        """Merge results from different search strategies and rank by relevance"""
        all_docs = []
        all_metas = []
        all_distances = []
        
        # Collect all results with source tracking
        sources = [
            (semantic_results, "semantic", 1.0),
            (keyword_results, "keyword", 0.8),
            (phrase_results, "phrase", 0.9)
        ]
        
        for results, source_type, weight in sources:
            if results.get("documents") and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    distance = results["distances"][0][i] if results.get("distances") and results["distances"][0] else 1.0
                    weighted_distance = distance / weight  # Lower is better
                    
                    all_docs.append(doc)
                    all_metas.append({
                        **results["metadatas"][0][i],
                        "search_type": source_type,
                        "weighted_distance": weighted_distance
                    })
                    all_distances.append(weighted_distance)
        
        # Remove duplicates while preserving best scores
        unique_results = {}
        for doc, meta, dist in zip(all_docs, all_metas, all_distances):
            doc_key = doc[:100]  # Use first 100 chars as key for deduplication
            if doc_key not in unique_results or dist < unique_results[doc_key][2]:
                unique_results[doc_key] = (doc, meta, dist)
        
        # Sort by weighted distance (lower is better) and limit results
        sorted_results = sorted(unique_results.values(), key=lambda x: x[2])[:n_results]
        
        # Format as ChromaDB result structure
        final_docs = [result[0] for result in sorted_results]
        final_metas = [result[1] for result in sorted_results]
        final_distances = [result[2] for result in sorted_results]
        
        return {
            "documents": [final_docs],
            "metadatas": [final_metas],
            "distances": [final_distances]
        }

    def _run(self, query: str, n_results: int = 15, collection_name: str = None, answer_choices: Optional[List[str]] = None) -> List[str]:
        """
        Enhanced query method with exact text search, query expansion, and answer choice analysis.
        """
        collection_to_query = self._collection

        if not collection_to_query:
            return ["❌ Error: No collection selected for query. Please select a collection at the start of the session."]

        # Debug output to show what query is being made
        print(f"\n🔍 ADVANCED RAG DEBUGGING:")
        print(f"   Query: '{query}'")
        print(f"   Collection: {collection_to_query.name}")
        print(f"   Requesting {n_results} results")
        print(f"   Answer choices provided: {answer_choices}")
        
        # Generate query variations
        expanded_queries = self._expand_query(query)
        print(f"   📝 Expanded queries: {expanded_queries}")
        
        # Extract key terms for debugging
        key_terms = self._extract_key_terms(query)
        print(f"   🔑 Key terms: {key_terms}")

        # Perform multiple search strategies with expanded queries
        all_results = []
        
        print(f"   🔄 Running exact text search...")
        exact_results = self._exact_text_search(query, n_results)
        if exact_results.get("documents") and exact_results["documents"][0]:
            print(f"      ✅ Exact search found {len(exact_results['documents'][0])} results")
        else:
            print(f"      ❌ Exact search found no results")
        all_results.append((exact_results, "exact", 1.2))
        
        print(f"   🔄 Running semantic searches...")
        for i, exp_query in enumerate(expanded_queries[:3]):  # Use top 3 expanded queries
            print(f"      🔍 Semantic search {i+1}: '{exp_query}'")
            semantic_results = self._semantic_search(exp_query, n_results)
            if semantic_results.get("documents") and semantic_results["documents"][0]:
                print(f"         ✅ Found {len(semantic_results['documents'][0])} results")
            else:
                print(f"         ❌ Found no results")
            weight = 1.0 - (i * 0.1)  # Decrease weight for each expansion
            all_results.append((semantic_results, f"semantic-{i+1}", weight))
        
        print(f"   🔄 Running keyword search...")
        keyword_results = self._keyword_search(query, n_results)
        if keyword_results.get("documents") and keyword_results["documents"][0]:
            print(f"      ✅ Keyword search found {len(keyword_results['documents'][0])} results")
        else:
            print(f"      ❌ Keyword search found no results")
        all_results.append((keyword_results, "keyword", 0.8))
        
        print(f"   🔄 Running phrase search...")
        phrase_results = self._phrase_search(query, n_results)
        if phrase_results.get("documents") and phrase_results["documents"][0]:
            print(f"      ✅ Phrase search found {len(phrase_results['documents'][0])} results")
        else:
            print(f"      ❌ Phrase search found no results")
        all_results.append((phrase_results, "phrase", 0.9))
        
        # Merge all results
        print(f"   🔄 Merging {len(all_results)} search strategies...")
        final_results = self._merge_multiple_results(all_results, n_results)
        
        # Show what was actually found in the final results
        if final_results.get("documents") and final_results["documents"][0]:
            print(f"   📋 FINAL MERGED RESULTS: {len(final_results['documents'][0])} documents")
            for i, doc in enumerate(final_results["documents"][0][:3]):  # Show first 3
                print(f"      {i+1}. {doc[:100]}...")
        else:
            print(f"   ❌ NO FINAL RESULTS AFTER MERGING")
        
        # Analyze answer choices if provided
        choice_analysis = {}
        if answer_choices:
            print(f"   🎯 ANALYZING ANSWER CHOICES:")
            choice_analysis = self._analyze_answer_choices(query, answer_choices, final_results)
            print(f"   📊 Raw choice scores: {choice_analysis}")
            
            # Show detailed analysis for each choice
            for choice, score in choice_analysis.items():
                print(f"      '{choice}': score={score:.2f}")
                # Check if this choice appears in any of the top documents
                found_in_docs = []
                if final_results.get("documents") and final_results["documents"][0]:
                    for i, doc in enumerate(final_results["documents"][0][:5]):
                        if choice.lower() in doc.lower():
                            found_in_docs.append(i+1)
                if found_in_docs:
                    print(f"         🎯 Found in documents: {found_in_docs}")
                else:
                    print(f"         ❌ Not found in any top documents")
        else:
            print(f"   ⚠️ NO ANSWER CHOICES PROVIDED - Cannot do choice analysis!")
        
        # Calculate confidence
        overall_confidence = self._calculate_confidence_score(final_results, query)
        print(f"   📊 Overall confidence: {overall_confidence:.2f}")
        
        # Check for potential issues
        if not final_results.get("documents") or not final_results["documents"][0]:
            print(f"   🚨 CRITICAL ISSUE: No documents found despite multiple search strategies!")
            print(f"   💡 SUGGESTIONS:")
            print(f"      - Check if the knowledge base contains relevant information")
            print(f"      - Try simpler search terms")
            print(f"      - Verify collection has been properly populated")
        elif not answer_choices:
            print(f"   ⚠️ WARNING: No answer choices provided - agent may struggle with selection")
        elif not choice_analysis or max(choice_analysis.values()) < 1.0:
            print(f"   ⚠️ WARNING: Low match scores for all answer choices - possible mismatch")
            print(f"   💡 Consider if answer choices match the document content style")
        
        docs_with_metadata = []
        if final_results.get("documents") and final_results["documents"][0]:
            # Add a summary header with confidence and answer analysis
            confidence_emoji = "🎯" if overall_confidence > 0.7 else "⚠️" if overall_confidence > 0.4 else "❓"
            docs_with_metadata.append(f"🔍 Found {len(final_results['documents'][0])} relevant documents for query: '{query}'")
            docs_with_metadata.append(f"{confidence_emoji} Overall confidence: {overall_confidence:.2f}")
            
            # Add answer choice recommendations if available
            if choice_analysis:
                sorted_choices = sorted(choice_analysis.items(), key=lambda x: x[1], reverse=True)
                best_choice = sorted_choices[0]
                
                if best_choice[1] > 0.5:  # Only recommend if score is reasonable
                    docs_with_metadata.append(f"🎯 RECOMMENDED ANSWER: '{best_choice[0]}' (score: {best_choice[1]:.2f})")
                else:
                    docs_with_metadata.append(f"⚠️ LOW CONFIDENCE RECOMMENDATION: '{best_choice[0]}' (score: {best_choice[1]:.2f})")
                    docs_with_metadata.append(f"💡 All choices have low scores - answer may not be explicitly stated")
                
                # Show all choice scores
                docs_with_metadata.append(f"📊 All answer choice scores:")
                for choice, score in sorted_choices:
                    emoji = "🎯" if score > 3 else "📍" if score > 1 else "📌"
                    docs_with_metadata.append(f"   {emoji} {choice}: {score:.2f}")
            else:
                docs_with_metadata.append(f"⚠️ NO ANSWER CHOICE ANALYSIS - Agent must interpret results manually")
            
            docs_with_metadata.append("="*80)
            
            # Show top results with more context
            result_limit = 5 if overall_confidence > 0.6 else 8
            
            for i in range(min(result_limit, len(final_results["documents"][0]))):
                doc_content = final_results["documents"][0][i]
                doc_metadata = final_results["metadatas"][0][i]
                distance = final_results["distances"][0][i] if final_results.get("distances") else "N/A"
                
                # Get metadata info
                page_info = doc_metadata.get('page', 'N/A')
                source_info = doc_metadata.get('source', 'unknown')
                search_type = doc_metadata.get('search_type', 'unknown')
                
                # Clean up the content formatting
                clean_content = doc_content.strip()
                
                # Add relevance indicator
                relevance = "🎯 HIGH" if distance < 0.3 else "📍 MEDIUM" if distance < 0.6 else "📌 LOW"
                
                # Highlight answer choice matches if available
                content_with_highlights = clean_content
                if answer_choices:
                    for choice in answer_choices:
                        # More flexible matching
                        choice_words = choice.lower().split()
                        for word in choice_words:
                            if len(word) > 3 and word in clean_content.lower():
                                # Highlight the word
                                clean_content = re.sub(
                                    f"\\b{re.escape(word)}\\b", 
                                    f">>> {word} <<<", 
                                    clean_content, 
                                    flags=re.IGNORECASE
                                )
                        
                        # Also check for exact phrase match
                        if choice.lower() in clean_content.lower():
                            content_with_highlights = clean_content.replace(
                                choice, f"🎯🎯🎯 {choice} 🎯🎯🎯"
                            )
                
                # Format the result with better structure
                result_text = f"""
📄 Document {i+1} ({relevance} relevance, via {search_type} search):
Content: {content_with_highlights}
Source: {source_info}
Page: {page_info}
{"-"*60}"""
                docs_with_metadata.append(result_text)
            
            # Add guidance based on confidence and choice analysis
            if choice_analysis:
                best_score = max(choice_analysis.values())
                if best_score > 5:
                    docs_with_metadata.append(f"\n🎯 HIGH MATCH CONFIDENCE: Strong evidence for recommended answer.")
                elif best_score > 2:
                    docs_with_metadata.append(f"\n📋 MEDIUM MATCH CONFIDENCE: Reasonable evidence found.")
                elif best_score > 0.5:
                    docs_with_metadata.append(f"\n⚠️ LOW MATCH CONFIDENCE: Weak evidence - answer may be inferred.")
                else:
                    docs_with_metadata.append(f"\n❓ VERY LOW CONFIDENCE: Answer may not be directly stated in documents.")
                    docs_with_metadata.append(f"   Consider if the question requires inference or the knowledge base is incomplete.")

        if not docs_with_metadata:
            error_msg = [f"❌ No relevant documents found for query: '{query}'."]
            error_msg.append(f"🔍 DEBUGGING INFO:")
            error_msg.append(f"   - Collection: {collection_to_query.name}")
            error_msg.append(f"   - Total documents in collection: {collection_to_query.count()}")
            error_msg.append(f"   - Expanded queries tried: {expanded_queries}")
            error_msg.append(f"   - Answer choices: {answer_choices}")
            error_msg.append(f"💡 SUGGESTIONS:")
            error_msg.append(f"   - Verify the knowledge base contains information about this topic")
            error_msg.append(f"   - Try broader or simpler search terms")
            error_msg.append(f"   - Check if documents were properly processed during embedding")
            return error_msg
        
        return docs_with_metadata

    def _merge_multiple_results(self, all_results: List[Tuple[Dict, str, float]], n_results: int) -> Dict:
        """Merge results from multiple search strategies with different weights"""
        all_docs = []
        all_metas = []
        all_distances = []
        
        for results, source_type, weight in all_results:
            if results.get("documents") and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    distance = results["distances"][0][i] if results.get("distances") and results["distances"][0] else 1.0
                    weighted_distance = distance / weight  # Lower is better
                    
                    all_docs.append(doc)
                    all_metas.append({
                        **results["metadatas"][0][i],
                        "search_type": source_type,
                        "weighted_distance": weighted_distance
                    })
                    all_distances.append(weighted_distance)
        
        # Remove duplicates while preserving best scores
        unique_results = {}
        for doc, meta, dist in zip(all_docs, all_metas, all_distances):
            doc_key = doc[:100]  # Use first 100 chars as key for deduplication
            if doc_key not in unique_results or dist < unique_results[doc_key][2]:
                unique_results[doc_key] = (doc, meta, dist)
        
        # Sort by weighted distance (lower is better) and limit results
        sorted_results = sorted(unique_results.values(), key=lambda x: x[2])[:n_results]
        
        # Format as ChromaDB result structure
        final_docs = [result[0] for result in sorted_results]
        final_metas = [result[1] for result in sorted_results]
        final_distances = [result[2] for result in sorted_results]
        
        return {
            "documents": [final_docs],
            "metadatas": [final_metas],
            "distances": [final_distances]
        }
