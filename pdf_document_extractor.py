import pdfplumber
from typing import List, Dict, Tuple
import os

class PDFDocumentExtractor:
    """
    Extract text from PDF documents using pdfplumber
    """
    
    def __init__(self, pdf_path: str):
        """
        Initialize PDF extractor
        
        Args:
            pdf_path: Path to PDF file
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        self.pdf_path = pdf_path
        self.pdf = pdfplumber.open(pdf_path)
        self.metadata = self._extract_metadata()
    
    def _extract_metadata(self) -> Dict:
        """Extract PDF metadata"""
        metadata = {
            "filename": os.path.basename(self.pdf_path),
            "filepath": self.pdf_path,
            "total_pages": len(self.pdf.pages),
            "pdf_metadata": self.pdf.metadata if self.pdf.metadata else {}
        }
        return metadata
    
    def extract_full_text(self) -> str:
        """
        Extract all text from PDF
        
        Returns:
            Full text content
        """
        full_text = ""
        
        for page_num, page in enumerate(self.pdf.pages, 1):
            text = page.extract_text()
            if text:
                # Add page marker for tracking
                full_text += f"\n[PAGE {page_num}]\n"
                full_text += text
        
        return full_text
    
    def extract_text_with_pages(self) -> List[Dict]:
        """
        Extract text from each page separately
        
        Returns:
            List of dicts with page content and metadata
        """
        pages_data = []
        
        for page_num, page in enumerate(self.pdf.pages, 1):
            text = page.extract_text()
            
            if text:
                pages_data.append({
                    "page_number": page_num,
                    "text": text,
                    "text_length": len(text),
                    "char_count": len(text),
                    "has_content": True
                })
            else:
                pages_data.append({
                    "page_number": page_num,
                    "text": "",
                    "text_length": 0,
                    "char_count": 0,
                    "has_content": False
                })
        
        return pages_data
    
    def extract_tables(self) -> List[Dict]:
        """
        Extract tables from PDF (if any)
        
        Returns:
            List of extracted tables with page info
        """
        tables = []
        
        for page_num, page in enumerate(self.pdf.pages, 1):
            page_tables = page.extract_tables()
            
            if page_tables:
                for table_idx, table in enumerate(page_tables):
                    tables.append({
                        "page_number": page_num,
                        "table_index": table_idx,
                        "rows": len(table),
                        "columns": len(table[0]) if table else 0,
                        "table_data": table
                    })
        
        return tables
    
    def get_page_text(self, page_number: int) -> str:
        """
        Get text from specific page
        
        Args:
            page_number: Page number (1-indexed)
        
        Returns:
            Text content of page
        """
        if page_number < 1 or page_number > len(self.pdf.pages):
            raise ValueError(f"Invalid page number: {page_number}")
        
        page = self.pdf.pages[page_number - 1]
        return page.extract_text()
    
    def get_statistics(self) -> Dict:
        """Get PDF statistics"""
        pages_data = self.extract_text_with_pages()
        
        total_chars = sum(p["char_count"] for p in pages_data)
        pages_with_content = sum(1 for p in pages_data if p["has_content"])
        
        return {
            "total_pages": self.metadata["total_pages"],
            "pages_with_content": pages_with_content,
            "total_characters": total_chars,
            "estimated_tokens": total_chars // 4,  # Rough estimate
            "average_chars_per_page": total_chars // pages_with_content if pages_with_content > 0 else 0
        }
    
    def close(self):
        """Close PDF file"""
        self.pdf.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Test PDF extraction
if __name__ == "__main__":
    pdf_path = r"D:\PersonalProjs\2SAF\US_TRADOC\test_data\ARN44767-FM_3-01-000-WEB-1.pdf"  
    
    print("=== Testing PDF Extraction ===")
    
    with PDFDocumentExtractor(pdf_path) as extractor:
        # Print metadata
        print(f"PDF: {extractor.metadata['filename']}")
        print(f"Pages: {extractor.metadata['total_pages']}")
        
        # Get statistics
        stats = extractor.get_statistics()
        print(f"\nStatistics:")
        print(f"  Total characters: {stats['total_characters']}")
        print(f"  Estimated tokens: {stats['estimated_tokens']}")
        print(f"  Pages with content: {stats['pages_with_content']}")
        
        # Extract first page
        first_page_text = extractor.get_page_text(1)
        print(f"\nFirst page preview (first 200 chars):")
        print(first_page_text[:200] + "...")