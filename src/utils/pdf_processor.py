import os
import io
import re
import PyPDF2
import traceback
from typing import Optional, Tuple

class PDFProcessor:
    """Utility class for processing PDF files."""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> Optional[str]:
        """Extract text content from a PDF file."""
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
            
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                
                # Check if PDF is encrypted
                if reader.is_encrypted:
                    try:
                        # Try to decrypt with empty password
                        reader.decrypt('')
                    except:
                        print(f"Error: PDF {file_path} is encrypted and could not be decrypted")
                        return None
                
                # Extract text from each page
                for page_num in range(len(reader.pages)):
                    try:
                        page = reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as page_error:
                        print(f"Error extracting text from page {page_num} of {file_path}: {str(page_error)}")
                        # Continue with next page instead of failing completely
                
                # Clean up the text
                text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
                text = text.strip()
                
                # Check if we got any text
                if not text:
                    print(f"Warning: No text extracted from {file_path}. The PDF might be scanned or image-based.")
                
                return text
        except Exception as e:
            print(f"Error extracting text from PDF {file_path}: {str(e)}")
            traceback.print_exc()  # Print full traceback for debugging
            return None