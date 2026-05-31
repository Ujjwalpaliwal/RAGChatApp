import os
import sys

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.ingestion.chunker import process_documents
from src.search.embedder import DocumentEmbedder
from src.search.vector_store import FAISSVectorStore

def build_index():
    data_dir = os.path.join(project_root, "data")
    
    # 1. Scan for supported documents
    if not os.path.exists(data_dir):
        print(f"Data directory not found at {data_dir}. Creating it.")
        os.makedirs(data_dir, exist_ok=True)
        return
        
    file_paths = []
    for file in os.listdir(data_dir):
        if file.endswith(('.txt', '.pdf', '.docx')):
            file_paths.append(os.path.join(data_dir, file))
            
    print(f"Found files to index: {[os.path.basename(f) for f in file_paths]}")
    
    if not file_paths:
        print("No supported files (.txt, .pdf, .docx) found inside the data/ folder.")
        return
        
    # 2. Process (parse + chunk)
    documents = process_documents(file_paths)
    print(f"Generated {len(documents)} text chunks from files.")
    
    if not documents:
        print("No content chunks could be created.")
        return
        
    # 3. Generate embeddings
    print("Loading embedding model...")
    embedder = DocumentEmbedder()
    texts = [doc['content'] for doc in documents]
    
    print("Generating dense vector embeddings...")
    embeddings = embedder.encode(texts, show_progress_bar=True)
    
    # 4. Save FAISS index
    print("Initializing FAISS Vector Store...")
    vector_store = FAISSVectorStore()
    vector_store.build_and_save(embeddings, documents)
    
    print("Successfully built and saved FAISS index!")

if __name__ == "__main__":
    build_index()