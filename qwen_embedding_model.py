from sentence_transformers import SentenceTransformer
from typing import List
import torch

class QwenEmbeddingModel:
    """
    Wrapper for Qwen3-Embedding-4B model using sentence-transformers
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-4B"):
        """
        Initialize Qwen embedding model
        
        Args:
            model_name: HuggingFace model identifier
        """
        # Check if GPU is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        self.embedding_dim = 2048  # Qwen3-Embedding-4B output dimension
        
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Embed a list of texts
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing (adjust based on GPU memory)
        
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()
    
    def embed_single(self, text: str) -> List[float]:
        """
        Embed a single text
        
        Args:
            text: Single text string to embed
        
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query (same as embed_single, but for clarity)
        
        Args:
            query: Query string to embed
        
        Returns:
            Embedding vector
        """
        return self.embed_single(query)

# Test the embedding model
if __name__ == "__main__":
    # Initialize
    embedding_model = QwenEmbeddingModel()
    
    # Test with sample texts
    test_texts = [
        "Personal information includes names and addresses.",
        "Organizations must collect data with consent.",
        "The weather is nice today."
    ]
    
    print("\n=== Testing Embedding Model ===")
    embeddings = embedding_model.embed_texts(test_texts)
    
    print(f"Input texts: {len(test_texts)}")
    print(f"Embeddings shape: {len(embeddings)}x{len(embeddings[0])}")
    
    # Calculate similarity between first two texts
    import numpy as np
    
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    sim_01 = cosine_similarity(embeddings[0], embeddings[1])
    sim_02 = cosine_similarity(embeddings[0], embeddings[2])
    
    print(f"\nSimilarity between text 0 and 1: {sim_01:.4f} (similar topics)")
    print(f"Similarity between text 0 and 2: {sim_02:.4f} (different topics)")