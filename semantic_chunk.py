import re
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from qwen_embedding_model import QwenEmbeddingModel

@dataclass
class SemanticChunk:
    """Data class for a semantic chunk"""
    id: str
    text: str
    sentences: List[str]
    page_numbers: List[int]
    start_position: int
    end_position: int
    token_count: int
    embedding: List[float]
    quality_score: float
    metadata: Dict

class SemanticChunker:
    """
    Semantic chunker using Qwen3-Embedding-4B
    """
    
    def __init__(self, embedding_model: QwenEmbeddingModel, 
                 similarity_threshold: float = 0.75,
                 min_chunk_length: int = 128):
        """
        Initialize semantic chunker
        
        Args:
            embedding_model: QwenEmbeddingModel instance
            similarity_threshold: Threshold for semantic similarity (0-1)
                                 Lower = split more often
                                 Higher = combine more chunks
            min_chunk_length: Minimum characters per chunk
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.min_chunk_length = min_chunk_length
    
    def split_into_sentences(self, text: str) -> List[Tuple[str, int]]:
        """
        Split text into sentences
        
        Returns:
            List of (sentence, character_position) tuples
        """
        # Enhanced sentence splitting
        # Handles: ".", "!", "?", abbreviations, etc.
        
        sentences = []
        position = 0
        
        # Split on sentence boundaries
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        splits = re.split(sentence_pattern, text)
        
        for sentence in splits:
            sentence = sentence.strip()
            if sentence:
                sentences.append((sentence, position))
                position += len(sentence)
        
        return sentences
    
    def _calculate_similarity(self, embedding1: List[float], 
                             embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Similarity score (0-1)
        """
        a = np.array(embedding1)
        b = np.array(embedding2)
        
        # Cosine similarity
        similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return float(similarity)
    
    def chunk_document(self, text: str, document_id: str = "doc_1",
                      source_file: str = None) -> List[SemanticChunk]:
        """
        Chunk document using semantic boundaries
        
        Args:
            text: Document text
            document_id: Identifier for document
            source_file: Original file path
        
        Returns:
            List of SemanticChunk objects
        """
        print("\n=== SEMANTIC CHUNKING ===")
        
        # Step 1: Split into sentences
        print("Step 1: Splitting into sentences...")
        sentences = self.split_into_sentences(text)
        print(f"  Created {len(sentences)} sentences")
        
        if len(sentences) < 2:
            # Document is too short, return as single chunk
            chunk = SemanticChunk(
                id=f"{document_id}_0",
                text=text,
                sentences=[s[0] for s in sentences],
                page_numbers=[1],
                start_position=0,
                end_position=len(text),
                token_count=len(text) // 4,
                embedding=self.embedding_model.embed_single(text),
                quality_score=0.8,
                metadata={"source_file": source_file}
            )
            return [chunk]
        
        # Step 2: Generate embeddings for each sentence
        print("Step 2: Generating embeddings...")
        sentence_texts = [s[0] for s in sentences]
        sentence_embeddings = self.embedding_model.embed_texts(sentence_texts)
        print(f"  Generated {len(sentence_embeddings)} embeddings")
        
        # Step 3: Calculate similarities between consecutive sentences
        print("Step 3: Calculating semantic similarities...")
        similarities = []
        for i in range(len(sentence_embeddings) - 1):
            sim = self._calculate_similarity(
                sentence_embeddings[i],
                sentence_embeddings[i + 1]
            )
            similarities.append(sim)
        
        # Step 4: Find split points (where similarity drops below threshold)
        print(f"Step 4: Finding split points (threshold: {self.similarity_threshold})...")
        split_indices = [0]  # Always start with first sentence
        
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                split_indices.append(i + 1)  # Split after sentence i
        
        split_indices.append(len(sentences))  # Always end with last sentence
        
        # Remove duplicates and sort
        split_indices = sorted(set(split_indices))
        print(f"  Found {len(split_indices) - 1} potential chunks")
        
        # Step 5: Create chunks
        print("Step 5: Creating chunks...")
        chunks = []
        chunk_id = 0
        
        for i in range(len(split_indices) - 1):
            start_idx = split_indices[i]
            end_idx = split_indices[i + 1]
            
            # Get sentences for this chunk
            chunk_sentences = sentence_texts[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)
            
            # Skip very small chunks
            if len(chunk_text) < self.min_chunk_length:
                continue
            
            # Get position info
            start_pos = sentences[start_idx][1]
            end_pos = sentences[end_idx - 1][1] + len(sentence_texts[end_idx - 1])
            
            # Average embedding for chunk
            chunk_embedding = np.mean(
                sentence_embeddings[start_idx:end_idx],
                axis=0
            ).tolist()
            
            # Calculate quality score
            quality = self._calculate_chunk_quality(chunk_text)
            
            chunk = SemanticChunk(
                id=f"{document_id}_chunk_{chunk_id}",
                text=chunk_text,
                sentences=chunk_sentences,
                page_numbers=[1],  # Will be updated if we have page info
                start_position=start_pos,
                end_position=end_pos,
                token_count=len(chunk_text) // 4,
                embedding=chunk_embedding,
                quality_score=quality,
                metadata={
                    "source_file": source_file,
                    "chunk_index": chunk_id,
                    "sentence_count": len(chunk_sentences),
                    "similarity_threshold": self.similarity_threshold
                }
            )
            
            chunks.append(chunk)
            chunk_id += 1
        
        print(f"Step 5 complete: Created {len(chunks)} final chunks")
        return chunks
    
    def _calculate_chunk_quality(self, text: str) -> float:
        """
        Calculate quality score for chunk (0-1)
        
        Factors:
        - Length (optimal: 200-500 chars)
        - Completeness (starts with capital, ends with punctuation)
        - Sentence count (minimum 2)
        """
        score = 0.0
        
        # Length check
        if 200 < len(text) < 500:
            score += 0.4
        elif len(text) > 100:
            score += 0.2
        
        # Completeness
        if text[0].isupper() and text[-1] in '.!?':
            score += 0.3
        elif text[0].isupper() or text[-1] in '.!?':
            score += 0.15
        
        # Sentence count
        sentences = text.split('. ')
        if len(sentences) >= 2:
            score += 0.3
        elif len(sentences) >= 1:
            score += 0.15
        
        return min(score, 1.0)

# Test the semantic chunker
if __name__ == "__main__":
    print("=== Testing Semantic Chunking ===\n")
    
    # Initialize embedding model
    print("Loading Qwen embedding model...")
    embedding_model = QwenEmbeddingModel()
    
    # Initialize chunker
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        similarity_threshold=0.75,
        min_chunk_length=128
    )
    
    # Sample text
    sample_text = """
    Personal information is any data that identifies an individual. This includes names, 
    addresses, phone numbers, and identification numbers. Organizations must handle personal 
    information carefully. Data collection must follow specific procedures. Consent is required 
    before collecting sensitive data. Storage requires encryption and security measures. 
    Access to personal information should be restricted. Data breaches must be reported immediately. 
    Individuals have the right to access their data. Corrections can be requested anytime.
    """
    
    # Chunk the document
    chunks = chunker.chunk_document(sample_text, document_id="test_doc")
    
    # Display results
    print("\n=== RESULTS ===")
    print(f"Total chunks: {len(chunks)}")
    print(f"Average quality: {np.mean([c.quality_score for c in chunks]):.2f}\n")
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}:")
        print(f"  Length: {len(chunk.text)} chars")
        print(f"  Quality: {chunk.quality_score:.2f}")
        print(f"  Text: {chunk.text[:100]}...\n")