import re
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """Clean and preprocess text by removing extra whitespaces and special characters."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:]', '', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [' '.join(words)]
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        
        if i + chunk_size >= len(words):
            break
            
    return chunks

def process_documents(file_paths: List[str], chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
    """Process a list of files into document chunk metadata."""
    from src.ingestion.parser import parse_document
    
    documents = []
    for file_path in file_paths:
        try:
            content = parse_document(file_path)
            cleaned_content = clean_text(content)
            chunks = chunk_text(cleaned_content, chunk_size, overlap)
            
            for i, chunk in enumerate(chunks):
                documents.append({
                    'content': chunk,
                    'source': file_path,
                    'chunk_id': i
                })
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            
    return documents
