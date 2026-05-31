import os
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any

class FAISSVectorStore:
    def __init__(self, index_path: str = None, doc_path: str = None):
        """Initialize and resolve FAISS index and documents metadata paths."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        
        self.index_path = index_path or os.path.join(project_root, "db", "faiss", "index.bin")
        self.doc_path = doc_path or os.path.join(project_root, "db", "faiss", "metadata.pkl")
        
        self.index = None
        self.documents = []
        
    def load(self) -> bool:
        """Load FAISS index and document metadata from disk if they exist."""
        if os.path.exists(self.index_path) and os.path.exists(self.doc_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.doc_path, 'rb') as f:
                self.documents = pickle.load(f)
            return True
        return False

    def build_and_save(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        """Create a new FAISS index, insert normalized embeddings, and save to disk."""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        
        # Normalize and add
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        self.documents = documents
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Save to disk
        faiss.write_index(self.index, self.index_path)
        with open(self.doc_path, 'wb') as f:
            pickle.dump(self.documents, f)
            
        print(f"FAISS index saved to {self.index_path}")
        print(f"Metadata saved to {self.doc_path}")

    def query(self, query_embedding: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        """Search the FAISS index for the top-k most similar chunks."""
        if self.index is None:
            raise ValueError("FAISS index is not loaded. Call load() or build_and_save() first.")
            
        # Copy to avoid modifying original query
        query_emb_copy = query_embedding.copy()
        faiss.normalize_L2(query_emb_copy)
        
        scores, indices = self.index.search(query_emb_copy.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['score'] = float(score)
                results.append(doc)
                
        return results
