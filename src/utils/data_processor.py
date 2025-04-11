import os
import pandas as pd
from typing import List, Dict, Any, Optional
from src.utils.pdf_processor import PDFProcessor

class DataProcessor:
    """Utility class for processing data files for the ResuRank system."""
    
    @staticmethod
    def load_job_descriptions_from_csv(csv_path: str) -> List[Dict[str, str]]:
        """Load job descriptions from a CSV file."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            job_descriptions = []
            
            for _, row in df.iterrows():
                if 'Job Title' in df.columns and 'Job Description' in df.columns:
                    job_descriptions.append({
                        'title': row['Job Title'],
                        'description': row['Job Description']
                    })
            
            return job_descriptions
        except Exception as e:
            raise Exception(f"Error loading job descriptions from CSV: {str(e)}")
    
    @staticmethod
    def load_resumes_from_folder(folder_path: str) -> List[Dict[str, Any]]:
        """Load resumes from a folder containing PDF files."""
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        resumes = []
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(folder_path, filename)
                
                try:
                    # Extract text from PDF
                    text = PDFProcessor.extract_text_from_pdf(file_path)
                    
                    if text:
                        resumes.append({
                            'filename': filename,
                            'text': text,
                            'path': file_path
                        })
                except Exception as e:
                    print(f"Error processing resume {filename}: {str(e)}")
        
        return resumes
    
    @staticmethod
    def get_resume_count_by_folder(base_folder: str) -> Dict[str, int]:
        """Get the count of resume files in each subfolder."""
        if not os.path.exists(base_folder):
            raise FileNotFoundError(f"Base folder not found: {base_folder}")
        
        folder_counts = {}
        
        for item in os.listdir(base_folder):
            item_path = os.path.join(base_folder, item)
            
            if os.path.isdir(item_path) and item.startswith('CV'):
                pdf_count = sum(1 for f in os.listdir(item_path) if f.lower().endswith('.pdf'))
                folder_counts[item] = pdf_count
        
        return folder_counts