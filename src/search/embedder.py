from sentence_transformers import SentenceTransformer
import numpy as np

class DocumentEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize SentenceTransformer model."""
        self.model = SentenceTransformer(model_name)
        
    def encode(self, texts: list, show_progress_bar: bool = False) -> np.ndarray:
        """Generate embeddings for the given list of texts."""
        return self.model.encode(texts, show_progress_bar=show_progress_bar)
