import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from app.utils import load_documents

class VectorIndexBuilder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
    
    def build_index(self, data_dir: str = "../data"):
        """Build FAISS index from documents in data directory."""
        # Get all text files in data directory
        file_paths = []
        for file in os.listdir(data_dir):
            if file.endswith('.txt'):
                file_paths.append(os.path.join(data_dir, file))
        
        print(f"Found {len(file_paths)} text files: {[os.path.basename(fp) for fp in file_paths]}")
        
        # Load and chunk documents
        self.documents = load_documents(file_paths)
        print(f"Created {len(self.documents)} document chunks")
        
        # Generate embeddings
        texts = [doc['content'] for doc in self.documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        print(f"Built FAISS index with {self.index.ntotal} vectors")
        
        return self.index, self.documents
    
    def save_index(self, index_path: str = "../embeddings/faiss_index.bin", 
                   doc_path: str = "../embeddings/faiss_index.pkl"):
        """Save FAISS index and documents."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save documents metadata
        with open(doc_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        print(f"Index saved to {index_path}")
        print(f"Documents metadata saved to {doc_path}")

if __name__ == "__main__":
    builder = VectorIndexBuilder()
    index, documents = builder.build_index()
    builder.save_index()