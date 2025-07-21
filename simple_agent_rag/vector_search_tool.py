import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from openai import OpenAI
import os
from sentence_transformers import SentenceTransformer


class VectorSearchTool:
    """Vector similarity search tool using OpenAI embeddings and FAISS"""
    
    def __init__(self, api_key: Optional[str] = None, embedding_model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.embedding_model = embedding_model
        self.dimension = 1536  # text-embedding-3-small dimension
        self.index = None
        self.documents = []
        self.metadata = []
        
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Fallback to sentence transformers if OpenAI fails
            fallback_model = SentenceTransformer('all-MiniLM-L6-v2')
            return fallback_model.encode([text])[0].astype(np.float32)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return np.vstack(embeddings)
    
    def build_index(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """Build FAISS index from documents"""
        print(f"Building vector index for {len(documents)} documents...")
        
        self.documents = documents
        self.metadata = metadata or [{"id": i} for i in range(len(documents))]
        
        # Generate embeddings
        embeddings = self.embed_texts(documents)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        print(f"Vector index built with {self.index.ntotal} documents")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Generate query embedding
        query_embedding = self.embed_text(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0:  # Valid index
                results.append({
                    "rank": i + 1,
                    "score": float(score),
                    "document": self.documents[idx],
                    "metadata": self.metadata[idx]
                })
        
        return results
    
    def save_index(self, filepath: str):
        """Save FAISS index to disk"""
        if self.index is not None:
            faiss.write_index(self.index, filepath)
    
    def load_index(self, filepath: str):
        """Load FAISS index from disk"""
        self.index = faiss.read_index(filepath)