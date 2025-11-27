import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class DocumentRetriever:
    def __init__(self, index_path: str = "../embeddings/faiss_index.bin", 
                 doc_path: str = "../embeddings/faiss_index.pkl",
                 model_name: str = 'all-MiniLM-L6-v2'):
        
        self.model = SentenceTransformer(model_name)
        
        # Load FAISS index
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        
        # Load documents metadata
        if os.path.exists(doc_path):
            with open(doc_path, 'rb') as f:
                self.documents = pickle.load(f)
        else:
            raise FileNotFoundError(f"Documents metadata not found at {doc_path}")
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant documents for the query."""
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Get relevant documents
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['score'] = float(score)
                results.append(doc)
        
        return results

    def get_context(self, query: str, k: int = 3) -> str:
        """Get formatted context from retrieved documents."""
        results = self.retrieve(query, k)
        
        if not results:
            return "No relevant information found."
        
        context_parts = []
        for i, doc in enumerate(results, 1):
            source_name = os.path.basename(doc['source'])
            context_parts.append(f"[Document {i} from {source_name}]:\n{doc['content']}")
        
        return "\n\n".join(context_parts)