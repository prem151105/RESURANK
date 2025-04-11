import os
import sys
import json
import csv
import PyPDF2
from src.agentnet_integration import AgentNetRecruitmentOrchestrator

def load_text_file(file_path):
    """Load text from a file with encoding fallbacks and special handling for CSV and PDF files."""
    # Special handling for PDF files
    if file_path.lower().endswith('.pdf'):
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error reading PDF file: {e}")
            raise
    
    # Special handling for CSV files
    if file_path.lower().endswith('.csv'):
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding, newline='') as file:
                    csv_reader = csv.DictReader(file)
                    # Get the job description from the first row
                    first_row = next(csv_reader)
                    return first_row.get('Job Description', '')
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error reading CSV with {encoding} encoding: {e}")
                continue
        raise Exception("Failed to read CSV file with any encoding")
    
    # Regular text file handling
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with {encoding} encoding: {e}")
            continue
    
    # If all encodings fail, try with errors='replace' as last resort
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            return file.read()
    except Exception as e:
        raise Exception(f"Failed to load file with any encoding: {e}")

def main():
    # Check if API key is set
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY environment variable not set.")
        print("Please set it using: set DEEPSEEK_API_KEY=your_api_key")
        return
    
    try:
        # Initialize the AgentNet-powered orchestrator
        orchestrator = AgentNetRecruitmentOrchestrator(api_key=api_key)
    except Exception as e:
        print(f"Error initializing orchestrator: {e}")
        print("Make sure all required packages are installed and the agentnet module is accessible")
        return
    
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python main.py <jd_file_path> <cv_folder_path>")
        return
    
    jd_file_path = sys.argv[1]
    cv_folder_path = sys.argv[2]
    
    # Load job description
    try:
        jd_text = load_text_file(jd_file_path)
    except Exception as e:
        print(f"Error loading job description file: {e}")
        return
    
    # Load CVs
    cv_texts = []
    try:
        # Get list of CV files
        cv_files = [f for f in os.listdir(cv_folder_path) if f.endswith('.txt') or f.endswith('.pdf')]
        if not cv_files:
            print("No CV files found in the specified folder.")
            return
        
        # Process all CV files
        for cv_file in cv_files:
            file_path = os.path.join(cv_folder_path, cv_file)
            print(f"Processing CV: {cv_file}")
            try:
                cv_text = load_text_file(file_path)
                cv_texts.append(cv_text)
            except Exception as e:
                print(f"Error processing CV {cv_file}: {e}")
                continue
        
    except Exception as e:
        print(f"Error loading CV files: {e}")
        return
    
    if not cv_texts:
        print("Failed to process any CV files.")
        return
    
    print(f"Loaded 1 job description and {len(cv_texts)} CVs.")
    
    # Run the full process
    print("Starting recruitment process...")
    try:
        result = orchestrator.run_full_process(jd_text, cv_texts)
        
        # Save results to a JSON file
        output_file = "recruitment_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        print(f"Process completed. Results saved to {output_file}")
        
        # Print summary
        print("\nSummary:")
        print(f"- Job Description processed: {len(result['jd_summary'])} key elements extracted")
        print(f"- Candidates processed: {len(result['candidates'])}")
        print(f"- Candidates shortlisted: {len(result['shortlisting']['shortlisted'])}")
        print(f"- Interviews scheduled: {len(result['scheduled_interviews'])}")
    except Exception as e:
        print(f"Error during recruitment process: {e}")
        print("Check your configuration and try again.")

if __name__ == "__main__":
    main()