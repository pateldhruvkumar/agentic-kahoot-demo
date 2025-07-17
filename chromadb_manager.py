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
        pdf_options.do_ocr = True  # Enable OCR for scanned documents
        pdf_options.do_table_structure = True  # Enable table structure recognition
        
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
            # Convert document using Docling
            result = self.converter.convert(file_path)

            # Extract different formats
            markdown_content = result.document.export_to_markdown()
            json_content = result.document.export_to_json()

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

            return {
                "content": markdown_content,
                "json_content": json_content,
                "metadata": metadata,
                "document_object": result.document
            }

        except Exception as e:
            print(f"❌ Docling parsing failed for '{file_path}': {e}. Please check the file format and try again.")
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
    
    def chunk_content(self, content: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Chunk content into smaller pieces for better retrieval
        
        Args:
            content: Text content to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
        """
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
    
    def embed_document(self, collection_name: str, file_path: str,
                      chunk_size: int = 1000, overlap: int = 100) -> bool:
        """
        Process a document with Docling and store it in ChromaDB.

        Args:
            collection_name (str): Name of the ChromaDB collection.
            file_path (str): Path to the document.
            chunk_size (int): Size of each text chunk for embedding.
            overlap (int): Overlap between chunks.

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

        # Chunk the content
        chunks = self.chunk_content(content, chunk_size, overlap)

        # Prepare data for ChromaDB
        documents = chunks
        ids = [f"{Path(file_path).stem}_{i}" for i in range(len(chunks))]
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk),
                "total_chunks": len(chunks)
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
            result = self.converter.convert(file_path)
            processing_time = time.time() - start_time
            
            # Extract structured data
            structured_data = {
                "title": self._extract_title(result.document),
                "headers": self._extract_headers(result.document),
                "tables": self._extract_tables(result.document),
                "images": self._extract_images(result.document),
                "references": self._extract_references(result.document),
                "metadata": {
                    "source": file_path,
                    "processing_time": processing_time,
                    "page_count": len(result.document.pages) if hasattr(result.document, 'pages') else 1
                }
            }
            
            # Update stats
            self.processing_stats["successful_extractions"] += 1
            self.processing_stats["total_processing_time"] += processing_time
            
            # Add main content to structured_data for chunking
            structured_data['content'] = result.document.export_to_markdown()

            return structured_data
            
        except Exception as e:
            self.processing_stats["failed_extractions"] += 1
            print(f"Error processing {file_path}: {e}")
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
            print(f"Document {file_path} already exists in collection")
            return False

        # Create different types of chunks
        chunks_data = []
        
        # Main content chunks
        main_content = structured_data.get('content', '')
        if main_content:
            main_chunks = self.chunk_content(main_content)
            for i, chunk in enumerate(main_chunks):
                chunks_data.append({
                    "content": chunk,
                    "type": "main_content",
                    "chunk_id": i,
                    "metadata": structured_data['metadata']
                })
        
        # Table chunks
        for i, table in enumerate(structured_data.get('tables', [])):
            chunks_data.append({
                "content": table['content'],
                "type": "table",
                "chunk_id": i,
                "metadata": {**structured_data['metadata'], **table}
            })
        
        # Header chunks
        for i, header in enumerate(structured_data.get('headers', [])):
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

        try:
            collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
            print(f"Successfully embedded {len(chunks_data)} structured chunks from {file_path}")
            return True
            
        except Exception as e:
            print(f"Failed to embed structured document: {e}")
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
