import json
from datetime import datetime
from typing import List, Dict, Tuple
from pdf_document_extractor import PDFDocumentExtractor
from qwen_embedding_model import QwenEmbeddingModel
import numpy as np
from semantic_chunk import SemanticChunker
from semantic_chunk import SemanticChunk

class PDFSemanticChunkingPipeline:
    """
    Complete pipeline for testing semantic chunking on a single PDF
    """
    
    def __init__(self, pdf_path: str):
        """
        Initialize pipeline
        
        Args:
            pdf_path: Path to test PDF
        """
        self.pdf_path = pdf_path
        self.embedding_model = None
        self.chunker = None
        self.pdf_text = None
        self.chunks = None
        self.results = {}
    
    def run_full_pipeline(self, similarity_threshold: float = 0.75,
                         save_results: bool = True) -> Dict:
        """
        Run complete pipeline
        
        Args:
            similarity_threshold: Semantic similarity threshold
            save_results: Whether to save results to file
        
        Returns:
            Dictionary with results
        """
        print("=" * 60)
        print("PDF SEMANTIC CHUNKING PIPELINE")
        print("=" * 60)
        
        # Step 1: Load PDF
        print("\n[1/4] Loading PDF...")
        self._load_pdf()
        
        # Step 2: Initialize models
        print("\n[2/4] Initializing models...")
        self._init_models(similarity_threshold)
        
        # Step 3: Chunk PDF
        print("\n[3/4] Chunking document...")
        self._chunk_document()
        
        # Step 4: Analyze results
        print("\n[4/4] Analyzing results...")
        self._analyze_results()
        
        # Save results
        if save_results:
            self._save_results()
        
        return self.results
    
    def _load_pdf(self):
        """Load and extract text from PDF"""
        with PDFDocumentExtractor(self.pdf_path) as extractor:
            print(f"  Filename: {extractor.metadata['filename']}")
            print(f"  Pages: {extractor.metadata['total_pages']}")
            
            stats = extractor.get_statistics()
            print(f"  Total characters: {stats['total_characters']:,}")
            print(f"  Estimated tokens: {stats['estimated_tokens']:,}")
            
            self.pdf_text = extractor.extract_full_text()
            self.results['pdf_metadata'] = {
                'filename': extractor.metadata['filename'],
                'pages': extractor.metadata['total_pages'],
                'characters': stats['total_characters'],
                'tokens_estimated': stats['estimated_tokens']
            }
    
    def _init_models(self, similarity_threshold: float):
        """Initialize embedding model and chunker"""
        print(f"  Loading Qwen3-Embedding-4B...")
        self.embedding_model = QwenEmbeddingModel()
        
        print(f"  Initializing chunker (threshold: {similarity_threshold})...")
        self.chunker = SemanticChunker(
            embedding_model=self.embedding_model,
            similarity_threshold=similarity_threshold,
            min_chunk_length=128
        )
        
        self.results['chunking_config'] = {
            'model': 'Qwen/Qwen3-Embedding-4B',
            'embedding_dimension': self.embedding_model.embedding_dim,
            'similarity_threshold': similarity_threshold
        }
    
    def _chunk_document(self):
        """Chunk the PDF text"""
        self.chunks = self.chunker.chunk_document(
            self.pdf_text,
            document_id=self.pdf_path.split('/')[-1].replace('.pdf', ''),
            source_file=self.pdf_path
        )
    
    def _analyze_results(self):
        """Analyze chunking results"""
        if not self.chunks:
            print("  No chunks created!")
            return
        
        # Calculate statistics
        chunk_lengths = [len(c.text) for c in self.chunks]
        chunk_tokens = [c.token_count for c in self.chunks]
        quality_scores = [c.quality_score for c in self.chunks]
        
        self.results['chunking_results'] = {
            'total_chunks': len(self.chunks),
            'chunk_statistics': {
                'min_chars': min(chunk_lengths),
                'max_chars': max(chunk_lengths),
                'avg_chars': int(np.mean(chunk_lengths)),
                'min_tokens': min(chunk_tokens),
                'max_tokens': max(chunk_tokens),
                'avg_tokens': int(np.mean(chunk_tokens))
            },
            'quality_statistics': {
                'avg_quality_score': float(np.mean(quality_scores)),
                'min_quality_score': float(min(quality_scores)),
                'max_quality_score': float(max(quality_scores)),
                'excellent_chunks': sum(1 for q in quality_scores if q > 0.8),
                'good_chunks': sum(1 for q in quality_scores if 0.6 <= q <= 0.8),
                'fair_chunks': sum(1 for q in quality_scores if q < 0.6)
            }
        }
        
        # Print results
        print(f"  Total chunks created: {len(self.chunks)}")
        print(f"  Average chunk size: {self.results['chunking_results']['chunk_statistics']['avg_chars']} chars")
        print(f"  Average quality score: {self.results['chunking_results']['quality_statistics']['avg_quality_score']:.2f}")
        print(f"  Excellent chunks (>0.8): {self.results['chunking_results']['quality_statistics']['excellent_chunks']}")
        print(f"  Good chunks (0.6-0.8): {self.results['chunking_results']['quality_statistics']['good_chunks']}")
    
    def _save_results(self):
        """Save results to JSON file"""
        output_file = f"chunking_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Prepare data for JSON (remove non-serializable objects)
        json_results = {
            'timestamp': datetime.now().isoformat(),
            'pdf_metadata': self.results['pdf_metadata'],
            'chunking_config': self.results['chunking_config'],
            'chunking_results': self.results['chunking_results'],
            'sample_chunks': [
                {
                    'id': self.chunks[0].id,
                    'text_preview': self.chunks[0].text[:200],
                    'length': len(self.chunks[0].text),
                    'quality_score': self.chunks[0].quality_score
                }
            ] if self.chunks else []
        }
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"  Results saved to: {output_file}")

# Main execution
if __name__ == "__main__":
    import sys
    
    # Check if PDF path provided
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Default test PDF
        pdf_path = r"D:\PersonalProjs\2SAF\US_TRADOC\test_data\ARN44767-FM_3-01-000-WEB-1.pdf"
    
    try:
        # Run pipeline
        pipeline = PDFSemanticChunkingPipeline(pdf_path)
        results = pipeline.run_full_pipeline(
            similarity_threshold=0.75,
            save_results=True
        )
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        
    except FileNotFoundError:
        print(f"Error: PDF file not found: {pdf_path}")
        print("\nUsage: python script.py <path_to_pdf>")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)