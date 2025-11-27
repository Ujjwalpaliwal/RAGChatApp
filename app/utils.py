import re
import numpy as np
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """Clean and preprocess text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:]', '', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
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

def load_documents(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Load and chunk documents from text files."""
    documents = []
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            cleaned_content = clean_text(content)
            chunks = chunk_text(cleaned_content)
            
            for i, chunk in enumerate(chunks):
                documents.append({
                    'content': chunk,
                    'source': file_path,
                    'chunk_id': i
                })
    
    return documents

def format_chat_history(history: List[Dict]) -> str:
    """Format chat history for context."""
    if not history:
        return ""
    
    formatted = []
    for msg in history[-6:]:  # Last 6 messages for context
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    
    return "\n".join(formatted)