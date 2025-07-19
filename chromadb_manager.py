"""
chromadb_manager.py

Docling + ChromaDB Knowledge Base Manager.

Environment Variables:
- OPENAI_API_KEY: API key for OpenAI embeddings (required for embedding functions)
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import sys
import time

# Docling imports
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption

# ChromaDB imports
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Used for embedding function

class ProgressIndicator:
    """Simple progress indicator for terminal"""
    
    def __init__(self, total_steps: int, task_name: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.task_name = task_name
        self.start_time = time.time()
    
    def update(self, step: int = None, message: str = None):
        """Update progress indicator"""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        percentage = (self.current_step / self.total_steps) * 100
        elapsed_time = time.time() - self.start_time
        
        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * self.current_step // self.total_steps)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Estimate remaining time
        if self.current_step > 0:
            estimated_total = elapsed_time * self.total_steps / self.current_step
            remaining_time = estimated_total - elapsed_time
            time_str = f" | ETA: {remaining_time:.0f}s"
        else:
            time_str = ""
        
        # Display message
        display_msg = f" - {message}" if message else ""
        
        # Print progress (overwrite previous line)
        sys.stdout.write(f"\r🔄 {self.task_name}: [{bar}] {percentage:.1f}% ({self.current_step}/{self.total_steps}){time_str}{display_msg}")
        sys.stdout.flush()
        
        if self.current_step >= self.total_steps:
            print(f"\n✅ {self.task_name} completed in {elapsed_time:.1f}s!")
    
    def finish(self, message: str = ""):
        """Mark as finished"""
        self.current_step = self.total_steps
        elapsed_time = time.time() - self.start_time
        print(f"\n✅ {self.task_name} completed in {elapsed_time:.1f}s! {message}")

class DoclingChromaProcessor:
    """
    A processor that uses Docling for document parsing and ChromaDB for storage.
    Handles document conversion, chunking, and embedding.
    """
    
    def __init__(self,
                 openai_api_key: str = None,
                 chroma_db_path: str = "./chroma_db",
                 embedding_model: str = "text-embedding-ada-002"):
        
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.chroma_db_path = chroma_db_path
        self.embedding_model = embedding_model
        
        # Initialize Docling converter
        self.converter = self._setup_docling_converter()
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
        
        # Setup embedding function
        self.embedding_function = self._setup_embedding_function()
    
    def _setup_docling_converter(self) -> DocumentConverter:
        """Setup Docling converter with optimal settings"""
        # Configure PDF pipeline options for better performance
        pdf_options = PdfPipelineOptions()
        pdf_options.do_ocr = False  # Disable OCR for large documents to save memory
        pdf_options.do_table_structure = False  # Disable table structure for large documents
        
        # Create converter with options
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
            }
        )
        
        return converter
    
    def _setup_embedding_function(self):
        """
        Setup OpenAI embedding function for ChromaDB.

        Raises:
            ValueError: If the OPENAI_API_KEY environment variable is not set.

        Returns:
            OpenAIEmbeddingFunction: The embedding function for ChromaDB.
        """
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your environment or .env file.")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.openai_api_key,
            model_name=self.embedding_model
        )

    def extract_text_with_docling(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text and metadata from a document using Docling.

        Args:
            file_path (str): Path to the document file.

        Returns:
            dict: Contains 'content' (markdown), 'json_content', 'metadata', and 'document_object'.
                  Returns None if extraction fails.

        Error Handling:
            Prints a clear, actionable error message if parsing fails.
        """
        try:
            # Show extraction progress
            print(f"📄 Starting text extraction for: {Path(file_path).name}")
            extraction_progress = ProgressIndicator(3, "Text Extraction")
            
            extraction_progress.update(1, "Converting PDF with Docling")
            # Convert document using Docling
            result = self.converter.convert(file_path)

            extraction_progress.update(2, "Exporting to markdown")
            # Extract different formats
            markdown_content = result.document.export_to_markdown()
            json_content = result.document.export_to_json()

            extraction_progress.update(3, "Processing metadata")
            # Extract metadata
            metadata = {
                "source": file_path,
                "title": getattr(result.document, 'title', Path(file_path).name),
                "page_count": len(result.document.pages) if hasattr(result.document, 'pages') else 1,
                "format": "markdown",
                "tables_count": self._count_tables(result.document),
                "images_count": self._count_images(result.document),
                "processing_time": getattr(result, 'processing_time', 0)
            }

            extraction_progress.finish(f"Extracted {len(markdown_content):,} characters")
            return {
                "content": markdown_content,
                "json_content": json_content,
                "metadata": metadata,
                "document_object": result.document
            }

        except Exception as e:
            print(f"\n❌ Docling parsing failed for '{file_path}': {e}. Please check the file format and try again.")
            return None
    
    def _count_tables(self, document) -> int:
        """Count tables in the document"""
        try:
            # This is a simplified count - adjust based on Docling's actual API
            return len([item for item in document.body.items if item.label == 'table'])
        except:
            return 0
    
    def _count_images(self, document) -> int:
        """Count images in the document"""
        try:
            return len([item for item in document.body.items if item.label == 'picture'])
        except:
            return 0

    def create_or_get_collection(self, collection_name: str):
        """Create or get existing collection using ChromaDB's built-in get_or_create_collection"""
        collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        print(f"Collection '{collection_name}' is ready (created or retrieved).")
        return collection
    
    def chunk_content(self, content: str, chunk_size: int = 1000, overlap: int = 100, use_semantic: bool = True) -> List[str]:
        """
        Chunk content into smaller pieces for better retrieval
        
        Args:
            content: Text content to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            use_semantic: Whether to use semantic chunking (recommended for better accuracy)
        """
        if use_semantic:
            return self._semantic_chunking(content, chunk_size, overlap)
        else:
            return self._fixed_size_chunking(content, chunk_size, overlap)
    
    def _semantic_chunking(self, content: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Implement semantic chunking that preserves context better than fixed-size chunking
        """
        # Try to use spaCy for better sentence detection if available
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            return self._spacy_semantic_chunking(content, chunk_size, overlap, nlp)
        except:
            print("⚠️ spaCy not available, using sentence-based semantic chunking")
            return self._sentence_based_chunking(content, chunk_size, overlap)
    
    def _spacy_semantic_chunking(self, content: str, chunk_size: int, overlap: int, nlp) -> List[str]:
        """Semantic chunking using spaCy for better sentence and entity detection"""
        doc = nlp(content)
        chunks = []
        current_chunk = ""
        current_entities = set()
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            
            # Extract entities in this sentence
            sent_entities = {ent.text.lower() for ent in sent.ents}
            
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sent_text if current_chunk else sent_text
            
            if len(potential_chunk) > chunk_size and current_chunk:
                # If there are shared entities, try to keep related sentences together
                entity_overlap = current_entities.intersection(sent_entities)
                
                if entity_overlap and len(potential_chunk) < chunk_size * 1.2:
                    # Allow slight size increase to keep related content together
                    current_chunk = potential_chunk
                    current_entities.update(sent_entities)
                else:
                    # Finalize current chunk and start new one
                    chunks.append(current_chunk.strip())
                    
                    # Add overlap from previous chunk
                    if overlap > 0 and chunks:
                        overlap_text = self._extract_overlap(chunks[-1], overlap)
                        current_chunk = overlap_text + " " + sent_text if overlap_text else sent_text
                    else:
                        current_chunk = sent_text
                    
                    current_entities = sent_entities
            else:
                current_chunk = potential_chunk
                current_entities.update(sent_entities)
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _sentence_based_chunking(self, content: str, chunk_size: int, overlap: int) -> List[str]:
        """Fallback semantic chunking using sentence boundaries"""
        # Split by sentences using multiple sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', content)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) > chunk_size and current_chunk:
                # Finalize current chunk
                chunks.append(current_chunk.strip())
                
                # Add overlap from previous chunk
                if overlap > 0 and chunks:
                    overlap_text = self._extract_overlap(chunks[-1], overlap)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk = potential_chunk
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _fixed_size_chunking(self, content: str, chunk_size: int, overlap: int) -> List[str]:
        """Original fixed-size chunking method (kept for compatibility)"""
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            
            # Try to break at sentence boundaries
            if end < len(content):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    chunk = content[start:break_point + 1]
                    end = break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
    
    def _extract_overlap(self, text: str, overlap_size: int) -> str:
        """Extract overlap text from the end of a chunk"""
        if len(text) <= overlap_size:
            return text
        
        # Try to break at sentence boundary within overlap region
        overlap_text = text[-overlap_size:]
        sentence_start = overlap_text.find('. ')
        
        if sentence_start > 0:
            return overlap_text[sentence_start + 2:]
        else:
            return overlap_text

    def embed_document(self, collection_name: str, file_path: str,
                      chunk_size: int = 600, overlap: int = 50, use_semantic_chunking: bool = True) -> bool:
        """
        Process a document with Docling and store it in ChromaDB.

        Args:
            collection_name (str): Name of the ChromaDB collection.
            file_path (str): Path to the document.
            chunk_size (int): Size of each text chunk for embedding (optimized for speed).
            overlap (int): Overlap between chunks (reduced for speed).
            use_semantic_chunking (bool): Use semantic chunking for better context preservation.

        Returns:
            bool: True if embedding was successful, False otherwise.

        Error Handling:
            Prints clear, actionable error messages for missing files, duplicate documents,
            extraction failures, and ChromaDB errors.
        """
        # Check if file exists
        if not os.path.isfile(file_path):
            print(f"❌ File not found: {file_path}. Please provide a valid file path.")
            return False

        # Get or create collection
        collection = self.create_or_get_collection(collection_name)

        # Check for existing document
        existing_docs = collection.get(where={"source": file_path})
        if existing_docs['ids']:
            print(f"ℹ️ Document '{file_path}' already exists in collection '{collection_name}'. Skipping embedding.")
            return False

        # Extract content using Docling
        extracted_data = self.extract_text_with_docling(file_path)
        if not extracted_data:
            print(f"❌ Failed to extract content from '{file_path}'. Ensure the file is a supported format and try again.")
            return False

        content = extracted_data['content']
        metadata = extracted_data['metadata']

        # Use enhanced chunking for better accuracy
        print(f"📚 Creating chunks using {'semantic' if use_semantic_chunking else 'fixed-size'} chunking...")
        chunks = self.chunk_content(content, chunk_size, overlap, use_semantic=use_semantic_chunking)
        print(f"✅ Created {len(chunks)} chunks (avg size: {sum(len(c) for c in chunks) // len(chunks)} chars)")

        # Prepare data for ChromaDB
        documents = chunks
        ids = [f"{Path(file_path).stem}_{i}" for i in range(len(chunks))]
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk),
                "total_chunks": len(chunks),
                "chunking_method": "semantic" if use_semantic_chunking else "fixed_size"
            })
            metadatas.append(chunk_metadata)

        # Add to collection
        try:
            collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
            print(f"✅ Successfully embedded {len(chunks)} chunks from '{file_path}' into collection '{collection_name}'.")
            return True
        except Exception as e:
            print(f"❌ Failed to add document to collection '{collection_name}': {e}")
            return False

    def embed_large_pdf(self, collection_name: str, file_path: str) -> bool:
        """
        Specialized method for embedding large PDF files (200+ pages)
        Uses memory-optimized processing and error recovery
        """
        import gc
        import os
        
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"📋 File size: {file_size_mb:.1f} MB")
        
        if file_size_mb > 50:
            print("⚠️  Large file detected. Using memory-optimized processing...")
            
        try:
            # Force garbage collection before processing
            gc.collect()
            
            # Use simplified Docling settings for large files
            simple_options = PdfPipelineOptions()
            simple_options.do_ocr = False
            simple_options.do_table_structure = False
            
            simple_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=simple_options)
                }
            )
            
            print("🔄 Converting PDF to text (this may take several minutes)...")
            conversion_progress = ProgressIndicator(2, "PDF Conversion")
            
            conversion_progress.update(1, "Processing with Docling")
            result = simple_converter.convert(file_path)
            
            conversion_progress.update(2, "Extracting text content")
            # Extract only markdown content (not JSON to save memory)
            content = result.document.export_to_markdown()
            conversion_progress.finish(f"Extracted {len(content):,} characters")
            
            # Small chunks for large documents
            chunk_size = 300  # Very small chunks
            overlap = 30
            print(f"\n📚 Creating small chunks for large document...")
            chunks = self.chunk_content(content, chunk_size=chunk_size, overlap=overlap)
            print(f"📦 Created {len(chunks)} small chunks (size: {chunk_size})")
            
            # Get collection
            collection = self.create_or_get_collection(collection_name)
            
            # Check for existing document
            existing_docs = collection.get(where={"source": file_path})
            if existing_docs['ids']:
                print(f"ℹ️ Document already exists. Skipping.")
                return False
            
            # Process in very small batches
            batch_size = 20
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            successful_batches = 0
            
            print(f"\n📤 Uploading {len(chunks)} chunks in {total_batches} small batches...")
            upload_progress = ProgressIndicator(total_batches, "Large File Upload")
            
            for i in range(0, len(chunks), batch_size):
                try:
                    batch_num = i // batch_size + 1
                    batch_chunks = chunks[i:i+batch_size]
                    batch_ids = [f"{Path(file_path).stem}_chunk_{j}" for j in range(i, i+len(batch_chunks))]
                    batch_metas = [{
                        "source": file_path,
                        "chunk_id": j,
                        "chunk_size": len(chunk),
                        "total_chunks": len(chunks)
                    } for j, chunk in enumerate(batch_chunks, i)]
                    
                    upload_progress.update(batch_num, f"Batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)")
                    
                    collection.add(
                        documents=batch_chunks,
                        ids=batch_ids,
                        metadatas=batch_metas
                    )
                    
                    successful_batches += 1
                    
                    # Longer delay for large files
                    time.sleep(1.0)
                    
                    # Force garbage collection every 10 batches
                    if (i // batch_size) % 10 == 0:
                        gc.collect()
                        
                except Exception as batch_error:
                    print(f"\n❌ Batch {batch_num} failed: {batch_error}")
                    continue
            
            upload_progress.finish(f"Processed {successful_batches}/{total_batches} batches successfully!")
            print(f"📊 Total chunks embedded: {successful_batches * batch_size}")
            return successful_batches > 0
            
        except MemoryError:
            print("\n❌ Out of memory! Try:")
            print("   1. Close other applications")
            print("   2. Split PDF into smaller files")
            print("   3. Use a machine with more RAM")
            return False
            
        except Exception as e:
            print(f"\n❌ Processing failed: {e}")
            print("💡 Suggestions:")
            print("   1. Check if PDF is corrupted")
            print("   2. Try splitting into smaller files")
            print("   3. Verify OpenAI API key and quota")
            return False

class AdvancedDoclingProcessor(DoclingChromaProcessor):
    """Advanced processor with additional features"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processing_stats = {
            "total_documents": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "total_processing_time": 0
        }
    
    def process_document_with_structure(self, file_path: str) -> Dict[str, Any]:
        """
        Process document and extract structured information
        """
        import time
        start_time = time.time()
        
        try:
            print(f"🔍 Analyzing document structure for: {Path(file_path).name}")
            structure_progress = ProgressIndicator(5, "Structure Analysis")
            
            structure_progress.update(1, "Converting document")
            result = self.converter.convert(file_path)
            processing_time = time.time() - start_time
            
            structure_progress.update(2, "Extracting title")
            title = self._extract_title(result.document)
            
            structure_progress.update(3, "Processing headers and sections")
            headers = self._extract_headers(result.document)
            
            structure_progress.update(4, "Extracting tables and images")
            tables = self._extract_tables(result.document)
            images = self._extract_images(result.document)
            references = self._extract_references(result.document)
            
            structure_progress.update(5, "Finalizing structured data")
            # Extract structured data
            structured_data = {
                "title": title,
                "headers": headers,
                "tables": tables,
                "images": images,
                "references": references,
                "metadata": {
                    "source": file_path,
                    "processing_time": processing_time,
                    "page_count": len(result.document.pages) if hasattr(result.document, 'pages') else 1
                }
            }
            
            # Add main content to structured_data for chunking
            structured_data['content'] = result.document.export_to_markdown()
            
            structure_progress.finish(f"Found {len(headers)} headers, {len(tables)} tables, {len(images)} images")
            
            # Update stats
            self.processing_stats["successful_extractions"] += 1
            self.processing_stats["total_processing_time"] += processing_time

            return structured_data
            
        except Exception as e:
            self.processing_stats["failed_extractions"] += 1
            print(f"\n❌ Error processing {Path(file_path).name}: {e}")
            return None
        
        finally:
            self.processing_stats["total_documents"] += 1
    
    def _extract_title(self, document) -> str:
        """Extract document title"""
        try:
            # Look for title in document structure
            for item in document.body.items:
                if item.label == 'title':
                    return item.text
            return "Untitled Document"
        except:
            return "Untitled Document"
    
    def _extract_headers(self, document) -> List[Dict]:
        """Extract all headers with hierarchy"""
        headers = []
        try:
            for item in document.body.items:
                if item.label.startswith('header'):
                    headers.append({
                        "text": item.text,
                        "level": item.label,
                        "page": getattr(item, 'page', 1)
                    })
        except:
            pass
        return headers
    
    def _extract_tables(self, document) -> List[Dict]:
        """Extract tables with structure"""
        tables = []
        try:
            for item in document.body.items:
                if item.label == 'table':
                    tables.append({
                        "content": item.text,
                        "page": getattr(item, 'page', 1),
                        "table_id": getattr(item, 'id', len(tables))
                    })
        except:
            pass
        return tables
    
    def _extract_images(self, document) -> List[Dict]:
        """Extract image information"""
        images = []
        try:
            for item in document.body.items:
                if item.label == 'picture':
                    images.append({
                        "caption": getattr(item, 'text', ''),
                        "page": getattr(item, 'page', 1),
                        "image_id": getattr(item, 'id', len(images))
                    })
        except:
            pass
        return images
    
    def _extract_references(self, document) -> List[str]:
        """Extract references/citations"""
        references = []
        try:
            for item in document.body.items:
                if item.label == 'reference':
                    references.append(item.text)
        except:
            pass
        return references
    
    def embed_structured_document(self, collection_name: str, file_path: str) -> bool:
        """
        Embed document with structured processing
        """
        # Process document
        structured_data = self.process_document_with_structure(file_path)
        if not structured_data:
            return False
        
        collection = self.create_or_get_collection(collection_name)
        
        # Check for existing document
        existing_docs = collection.get(where={"source": file_path})
        if existing_docs['ids']:
            print(f"ℹ️ Document {Path(file_path).name} already exists in collection")
            return False

        print(f"\n📚 Creating chunks for embedding...")
        
        # Create different types of chunks
        chunks_data = []
        
        # Main content chunks - REDUCED SIZE FOR LARGE DOCS
        main_content = structured_data.get('content', '')
        if main_content:
            # Use smaller chunks for large documents
            chunk_size = 500 if len(main_content) > 100000 else 1000
            main_chunks = self.chunk_content(main_content, chunk_size=chunk_size, overlap=50)
            print(f"📄 Created {len(main_chunks)} main content chunks (size: {chunk_size})")
            
            for i, chunk in enumerate(main_chunks):
                chunks_data.append({
                    "content": chunk,
                    "type": "main_content",
                    "chunk_id": i,
                    "metadata": structured_data['metadata']
                })
        
        # Table chunks
        tables = structured_data.get('tables', [])
        if tables:
            print(f"📊 Adding {len(tables)} table chunks")
            for i, table in enumerate(tables):
                chunks_data.append({
                    "content": table['content'],
                    "type": "table",
                    "chunk_id": i,
                    "metadata": {**structured_data['metadata'], **table}
                })
        
        # Header chunks
        headers = structured_data.get('headers', [])
        if headers:
            print(f"📋 Adding {len(headers)} header chunks")
            for i, header in enumerate(headers):
                chunks_data.append({
                    "content": header['text'],
                    "type": "header",
                    "chunk_id": i,
                    "metadata": {**structured_data['metadata'], **header}
                })
        
        # Prepare for ChromaDB
        documents = [item['content'] for item in chunks_data]
        ids = [f"{Path(file_path).stem}_{item['type']}_{item['chunk_id']}" for item in chunks_data]
        metadatas = [item['metadata'].copy() for item in chunks_data] # Create a copy to avoid modifying original
        
        # Ensure 'source' is correctly set in metadata for all chunks
        for meta in metadatas:
            meta['source'] = file_path # Override with current file_path

        # BATCH PROCESSING FOR LARGE DOCUMENTS
        batch_size = 50  # Process in smaller batches
        total_chunks = len(chunks_data)
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        print(f"\n📤 Uploading {total_chunks} chunks in {total_batches} batches...")
        upload_progress = ProgressIndicator(total_batches, "Embedding Upload")
        
        try:
            for i in range(0, total_chunks, batch_size):
                batch_num = i // batch_size + 1
                batch_end = min(i + batch_size, total_chunks)
                batch_docs = documents[i:batch_end]
                batch_ids = ids[i:batch_end]
                batch_metas = metadatas[i:batch_end]
                
                upload_progress.update(batch_num, f"Batch {batch_num}/{total_batches} ({len(batch_docs)} chunks)")
                
                collection.add(
                    documents=batch_docs,
                    ids=batch_ids,
                    metadatas=batch_metas
                )
                
                # Add small delay to avoid rate limits
                time.sleep(0.5)
            
            upload_progress.finish(f"Successfully embedded {len(chunks_data)} chunks!")
            return True
            
        except Exception as e:
            print(f"\n❌ Failed to embed structured document: {e}")
            print(f"💡 Try reducing chunk size or splitting the PDF into smaller files")
            return False

    def query_collection(self, collection_name: str, query: str,
                        n_results: int = 5, filter_dict: Dict = None) -> Dict:
        """
        Query collection with advanced filtering
        
        Args:
            collection_name: Name of collection to query
            query: Search query
            n_results: Number of results to return
            filter_dict: Filter criteria
        """
        try:
            collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            
            # Build query parameters
            query_params = {
                "query_texts": [query],
                "n_results": n_results
            }
            
            if filter_dict:
                query_params["where"] = filter_dict
            
            # Execute query
            results = collection.query(**query_params)
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]: # Check if documents list is not empty
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i],
                        "id": results['ids'][0][i]
                    })
            
            return {
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results)
            }
            
        except Exception as e:
            print(f"Query failed: {e}")
            return {"query": query, "results": [], "total_results": 0}
    
    def get_collection_stats(self, collection_name: str) -> Dict:
        """Get statistics about a collection"""
        try:
            collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            
            # Get all documents
            all_docs = collection.get()
            
            # Calculate statistics
            stats = {
                "total_documents": len(all_docs['ids']),
                "unique_sources": len(set(
                    meta.get('source', 'unknown')
                    for meta in all_docs['metadatas']
                )),
                "document_types": self._count_document_types(all_docs['metadatas']),
                "avg_chunk_size": self._calculate_avg_chunk_size(all_docs['documents']),
                "collection_name": collection_name
            }
            
            return stats
            
        except Exception as e:
            print(f"Failed to get collection stats: {e}")
            return {}
    
    def _count_document_types(self, metadatas: List[Dict]) -> Dict:
        """Count different document types"""
        type_counts = {}
        for meta in metadatas:
            doc_type = meta.get('type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        return type_counts
    
    def _calculate_avg_chunk_size(self, documents: List[str]) -> float:
        """Calculate average chunk size"""
        if not documents:
            return 0
        return sum(len(doc) for doc in documents) / len(documents)

class InteractiveKnowledgeBase:
    def __init__(self):
        self.processor = AdvancedDoclingProcessor()
        self.current_collection = None # Initialize current_collection to None
    
    def run(self):
        """Run interactive knowledge base manager"""
        print("=== Docling + ChromaDB Knowledge Base Manager ===")
        
        while True:
            print("\nOptions:")
            print("1. List collections")
            print("2. Create/Select collection")
            print("3. Add document")
            print("4. Query collection")
            print("5. View collection stats")
            print("6. Delete collection")
            print("7. Exit")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == "1":
                self.list_collections()
            elif choice == "2":
                self.create_or_select_collection()
            elif choice == "3":
                self.add_document()
            elif choice == "4":
                self.query_collection()
            elif choice == "5":
                self.view_stats()
            elif choice == "6":
                self.delete_collection()
            elif choice == "7":
                print("Goodbye!")
                break
            else:
                print("Invalid option")
    
    def list_collections(self):
        """List all collections"""
        collections = self.processor.chroma_client.list_collections()
        if collections:
            print("\nAvailable collections:")
            for i, collection in enumerate(collections, 1):
                print(f"{i}. {collection.name}")
        else:
            print("No collections found")
    
    def create_or_select_collection(self):
        """Create or select a collection"""
        name = input("Enter collection name: ").strip()
        if name:
            try:
                # Try to get the collection, if it exists, select it
                self.processor.create_or_get_collection(name)
                self.current_collection = name
                print(f"Selected collection: {name}")
            except Exception as e:
                print(f"Error creating/selecting collection: {e}")
    
    def add_document(self):
        """Add a document to current collection"""
        if not self.current_collection:
            print("Please select a collection first")
            return
        
        file_path = input("Enter file path: ").strip()
        if os.path.isfile(file_path):
            # Check file size to suggest appropriate method
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            if file_size_mb > 20:  # If larger than 20MB
                print(f"📋 Large file detected ({file_size_mb:.1f} MB)")
                print("Choose processing method:")
                print("1. Standard processing (may fail for very large files)")
                print("2. Large file processing (memory-optimized)")
                
                method = input("Select method (1 or 2): ").strip()
                
                if method == "2":
                    success = self.processor.embed_large_pdf(
                        collection_name=self.current_collection,
                        file_path=file_path
                    )
                else:
                    success = self.processor.embed_structured_document(
                        collection_name=self.current_collection,
                        file_path=file_path
                    )
            else:
                success = self.processor.embed_structured_document(
                    collection_name=self.current_collection,
                    file_path=file_path
                )
                
            if success:
                print("Document added successfully!")
            else:
                print("Failed to add document")
        else:
            print("File not found")
    
    def query_collection(self):
        """Query the current collection"""
        if not self.current_collection:
            print("Please select a collection first")
            return
        
        query = input("Enter your query: ").strip()
        if query:
            results = self.processor.query_collection(
                collection_name=self.current_collection,
                query=query,
                n_results=3
            )
            
            print(f"\nFound {results['total_results']} results:")
            for i, result in enumerate(results['results'], 1):
                print(f"\n--- Result {i} ---")
                print(f"Content: {result['content'][:300]}...")
                print(f"Source: {result['metadata'].get('source', 'unknown')}")
                print(f"Type: {result['metadata'].get('type', 'unknown')}")
                print(f"Similarity: {1 - result['distance']:.3f}")
    
    def view_stats(self):
        """View collection statistics"""
        if not self.current_collection:
            print("Please select a collection first")
            return
        
        stats = self.processor.get_collection_stats(self.current_collection)
        print(f"\n=== Collection Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    def delete_collection(self):
        """Delete a collection"""
        self.list_collections()
        name = input("Enter collection name to delete: ").strip()
        if name:
            confirm = input(f"Are you sure you want to delete '{name}'? (y/N): ")
            if confirm.lower() == 'y':
                try:
                    self.processor.chroma_client.delete_collection(name)
                    print(f"Collection '{name}' deleted")
                    if self.current_collection == name:
                        self.current_collection = None # Deselect if current collection is deleted
                except Exception as e:
                    print(f"Error deleting collection: {e}")

if __name__ == "__main__":
    manager = InteractiveKnowledgeBase()
    manager.run()
