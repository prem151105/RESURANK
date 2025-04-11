import streamlit as st
import os
import json
import tempfile
import shutil
import pandas as pd
from pathlib import Path
from src.agentnet_integration import AgentNetRecruitmentOrchestrator
from src.utils.pdf_processor import PDFProcessor
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

# Set page config
st.set_page_config(
    page_title="ResuRank - AI-Powered Recruitment System",
    page_icon="👔",
    layout="wide"
)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_text_file(file_path):
    """Load text from a file with encoding fallbacks."""
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

def extract_text(file):
    """Extract text from uploaded file based on its type"""
    try:
        if file.name.lower().endswith('.pdf'):
            # Save PDF temporarily to process it
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, file.name)
            with open(temp_path, 'wb') as f:
                f.write(file.getvalue())
            text = PDFProcessor.extract_text_from_pdf(temp_path)
            shutil.rmtree(temp_dir)  # Clean up
            if not text or text.strip() == "":
                return f"[Error: Could not extract text from {file.name}]"
            return text
        else:  # For txt files
            # Try different encodings if UTF-8 fails
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']
            
            for encoding in encodings_to_try:
                try:
                    return file.getvalue().decode(encoding)
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Error with {encoding} encoding: {e}")
                    continue
            
            # If all encodings fail, use 'replace' error handler as last resort
            return file.getvalue().decode('utf-8', errors='replace')
    except Exception as e:
        st.error(f"Error extracting text from {file.name}: {str(e)}")
        return f"[Error: {str(e)}]"

def process_files(jd_file, cv_files):
    """Process job description and CVs using the orchestrator"""
    # Check API key
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        st.error("DEEPSEEK_API_KEY environment variable not set. Please set it before running the application.")
        st.info("You can set it using: set DEEPSEEK_API_KEY=your_api_key")
        return None
    
    try:
        # Extract text from job description CSV
        jd_text = None
        if jd_file.name.lower().endswith('.csv'):
            try:
                # Try to read the file content directly
                file_content = jd_file.getvalue()
                
                # Try different encodings
                encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1', 'utf-16', 'ascii']
                decoded_content = None
                
                for encoding in encodings_to_try:
                    try:
                        decoded_content = file_content.decode(encoding)
                        st.success(f"Successfully decoded CSV with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if decoded_content is None:
                    # If all encodings fail, use 'replace' error handler
                    decoded_content = file_content.decode('utf-8', errors='replace')
                    st.warning("Using 'replace' error handler for decoding CSV")
                
                # Create a StringIO object to use with pandas
                import io
                csv_io = io.StringIO(decoded_content)
                
                # Read CSV from the StringIO object
                jd_df = pd.read_csv(csv_io)
                
                # Check if 'Job Description' column exists
                if 'Job Description' in jd_df.columns:
                    jd_text = jd_df.iloc[0]['Job Description']
                else:
                    # Try to find a column that might contain the job description
                    text_columns = [col for col in jd_df.columns if 'job' in col.lower() or 'description' in col.lower()]
                    if text_columns:
                        jd_text = jd_df.iloc[0][text_columns[0]]
                    else:
                        # Fallback to first non-empty column
                        for col in jd_df.columns:
                            if pd.notna(jd_df.iloc[0][col]) and str(jd_df.iloc[0][col]).strip():
                                jd_text = jd_df.iloc[0][col]
                                break
                
                # Clean up the text
                if jd_text is not None:
                    jd_text = str(jd_text).strip()
                    if not jd_text:
                        st.error("Job description text is empty. Please check the CSV file.")
                        return None
                else:
                    st.error("Could not find job description text in the CSV file.")
                    st.info("Please make sure the CSV file has a column containing the job description.")
                    return None
                    
            except Exception as e:
                st.error(f"Error loading job description: {e}")
                st.info("Please make sure the CSV file is properly formatted and contains the job description.")
                return None
        else:
            jd_text = extract_text(jd_file)
        
        # Validate job description
        if not jd_text or not isinstance(jd_text, str) or jd_text.strip() == "":
            st.error("Invalid job description text. Please check the job description file.")
            return None
        
        # Extract text from CVs
        cv_texts = []
        cv_names = []
        for cv_file in cv_files:
            try:
                cv_text = extract_text(cv_file)
                if cv_text and not cv_text.startswith("[Error:"):
                    cv_texts.append(cv_text)
                    cv_names.append(cv_file.name)
                    st.success(f"Successfully processed {cv_file.name}")
                else:
                    st.warning(f"Could not extract text from {cv_file.name}")
            except Exception as e:
                st.error(f"Error processing {cv_file.name}: {str(e)}")
        
        if not cv_texts:
            st.error("No valid CV texts to process. Please check the CV files.")
            return None
        
        # Initialize the orchestrator
        orchestrator = AgentNetRecruitmentOrchestrator(api_key=api_key)
        
        # Run the process
        with st.spinner("Processing files... This may take a few minutes."):
            result = orchestrator.run_full_process(jd_text, cv_texts)
        
        # Add source information if available
        if cv_names:
            for i, candidate in enumerate(result['candidates']):
                if i < len(cv_names):
                    candidate['source_file'] = cv_names[i]
        
        return result
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        st.info("Please check that your API key is valid and that the input files are properly formatted.")
        return None

def display_results(result):
    """Display the processing results in a structured format"""
    st.success("Processing completed successfully!")
    
    # Job Description Summary
    st.header("Job Description Summary")
    jd_summary = result.get('jd_summary', {})
    if jd_summary:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Key Requirements")
            for req in jd_summary.get('requirements', []):
                st.write(f"- {req}")
        with col2:
            st.subheader("Key Responsibilities")
            for resp in jd_summary.get('responsibilities', []):
                st.write(f"- {resp}")
    
    # Candidates Overview
    st.header("Candidates Overview")
    candidates = result.get('candidates', [])
    
    # Create a dataframe for better display
    candidate_data = []
    for i, candidate in enumerate(candidates):
        candidate_data.append({
            "ID": i+1,
            "Name": candidate.get('name', f"Candidate {i+1}"),
            "Source": candidate.get('source_file', 'Unknown'),
            "Match Score": f"{candidate.get('match_score', 0):.2f}%",
            "Skills Match": f"{candidate.get('skills_match', 0):.2f}%",
            "Experience Match": f"{candidate.get('experience_match', 0):.2f}%"
        })
    
    if candidate_data:
        df = pd.DataFrame(candidate_data)
        st.dataframe(df, use_container_width=True)
    
    # Shortlisted Candidates
    st.header("Shortlisted Candidates")
    shortlisted = result.get('shortlisting', {}).get('shortlisted', [])
    
    if shortlisted:
        for i, candidate_id in enumerate(shortlisted):
            if candidate_id < len(candidates):
                candidate = candidates[candidate_id]
                with st.expander(f"{i+1}. {candidate.get('name', f'Candidate {candidate_id+1}')} - Match: {candidate.get('match_score', 0):.2f}%"):
                    st.subheader("Skills")
                    for skill in candidate.get('skills', []):
                        st.write(f"- {skill}")
                    
                    st.subheader("Experience")
                    for exp in candidate.get('experience', []):
                        st.write(f"- {exp}")
                    
                    st.subheader("Education")
                    for edu in candidate.get('education', []):
                        st.write(f"- {edu}")
                    
                    st.subheader("Strengths")
                    for strength in candidate.get('strengths', []):
                        st.write(f"- {strength}")
                    
                    st.subheader("Weaknesses")
                    for weakness in candidate.get('weaknesses', []):
                        st.write(f"- {weakness}")
    else:
        st.info("No candidates were shortlisted.")
    
    # Interview Schedule
    st.header("Interview Schedule")
    interviews = result.get('scheduled_interviews', [])
    
    if interviews:
        for i, interview in enumerate(interviews):
            candidate_id = interview.get('candidate_id', 0)
            if candidate_id < len(candidates):
                candidate = candidates[candidate_id]
                st.write(f"{i+1}. {candidate.get('name', f'Candidate {candidate_id+1}')} - {interview.get('date', 'TBD')} at {interview.get('time', 'TBD')}")
    else:
        st.info("No interviews have been scheduled.")
    
    # Download option
    json_str = json.dumps(result, indent=2)
    st.download_button(
        label="Download Results as JSON",
        data=json_str,
        file_name="recruitment_results.json",
        mime="application/json"
    )

def load_results():
    try:
        with open('recruitment_results.json', 'r') as f:
            return json.load(f)
    except:
        return None

def plot_agent_architecture():
    """Create a visualization of the AgentNet architecture"""
    G = nx.DiGraph()
    
    # Add nodes with different colors for different agent types
    nodes = {
        "Coordinator": {"color": "lightblue", "type": "control"},
        "JD Summarizer": {"color": "lightgreen", "type": "processor"},
        "CV Processor": {"color": "lightgreen", "type": "processor"},
        "Shortlisting Agent": {"color": "lightgreen", "type": "processor"},
        "Scheduler": {"color": "lightgreen", "type": "processor"},
        "Memory": {"color": "lightyellow", "type": "storage"},
        "API Gateway": {"color": "lightpink", "type": "external"}
    }
    
    # Add nodes
    for node, props in nodes.items():
        G.add_node(node, **props)
    
    # Add edges with different styles
    edges = [
        # Control flow
        ("Coordinator", "JD Summarizer", {"style": "dashed"}),
        ("Coordinator", "CV Processor", {"style": "dashed"}),
        ("Coordinator", "Shortlisting Agent", {"style": "dashed"}),
        ("Coordinator", "Scheduler", {"style": "dashed"}),
        
        # Data flow
        ("JD Summarizer", "Memory", {"style": "solid"}),
        ("CV Processor", "Memory", {"style": "solid"}),
        ("Shortlisting Agent", "Memory", {"style": "solid"}),
        ("Scheduler", "Memory", {"style": "solid"}),
        
        # API connections
        ("JD Summarizer", "API Gateway", {"style": "dotted"}),
        ("CV Processor", "API Gateway", {"style": "dotted"}),
        ("Shortlisting Agent", "API Gateway", {"style": "dotted"}),
        ("Scheduler", "API Gateway", {"style": "dotted"})
    ]
    
    # Add edges
    for source, target, props in edges:
        G.add_edge(source, target, **props)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes
    node_colors = [G.nodes[node]["color"] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000)
    
    # Draw edges with different styles
    edge_styles = {
        "dashed": "--",
        "solid": "-",
        "dotted": ":"
    }
    
    for style in edge_styles:
        edges = [(u, v) for (u, v, d) in G.edges(data=True) if d.get("style") == style]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color="gray", style=edge_styles[style])
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='gray', linestyle='--', label='Control Flow'),
        plt.Line2D([0], [0], color='gray', linestyle='-', label='Data Flow'),
        plt.Line2D([0], [0], color='gray', linestyle=':', label='API Connection')
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.title("AgentNet Architecture", pad=20)
    plt.tight_layout()
    
    return plt

def plot_candidate_distribution(results):
    if not results or 'candidates' not in results:
        return None
        
    try:
        # Extract scores and filter out any None or invalid values
        scores = []
        for c in results['candidates']:
            match_score = c.get('match_score')
            if match_score is not None:
                try:
                    # Handle both string and numeric match scores
                    if isinstance(match_score, str):
                        # Remove any non-numeric characters and convert to float
                        clean_score = ''.join(c for c in match_score if c.isdigit() or c == '.')
                        if clean_score:
                            scores.append(float(clean_score))
                    else:
                        scores.append(float(match_score))
                except (ValueError, TypeError):
                    # Skip invalid scores
                    continue
        
        if not scores:
            return None
            
        plt.figure(figsize=(10, 6))
        sns.histplot(data=scores, bins=20)
        plt.title('Distribution of Candidate Match Scores')
        plt.xlabel('Match Score')
        plt.ylabel('Count')
        return plt
    except Exception as e:
        st.error(f"Error creating candidate distribution plot: {str(e)}")
        return None

def plot_shortlisting_metrics(results):
    if not results or 'shortlisting' not in results:
        return None
        
    try:
        shortlisted = len(results['shortlisting'].get('shortlisted', []))
        total = len(results.get('candidates', []))
        
        if total == 0:
            return None
            
        plt.figure(figsize=(8, 8))
        plt.pie([shortlisted, total-shortlisted], 
                labels=['Shortlisted', 'Not Shortlisted'],
                autopct='%1.1f%%',
                colors=['lightgreen', 'lightgray'])
        plt.title('Shortlisting Results')
        return plt
    except Exception as e:
        st.error(f"Error creating shortlisting metrics plot: {str(e)}")
        return None

def main():
    st.title("👔 ResuRank - AI-Powered Recruitment System")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "System Architecture", "Results Analysis"])
    
    if page == "Dashboard":
        st.header("Recruitment Dashboard")
        
        # File upload section
        st.subheader("Upload Files")
        jd_file = st.file_uploader("Upload Job Description (CSV)", type=['csv'])
        cv_files = st.file_uploader("Upload CVs (PDF)", type=['pdf'], accept_multiple_files=True)
        
        if st.button("Process Recruitment"):
            if jd_file and cv_files:
                with st.spinner("Processing..."):
                    # Process files
                    try:
                        results = process_files(jd_file, cv_files)
                        if results:
                            st.session_state.results = results
                            st.success("Processing completed!")
                            display_results(results)
                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
            else:
                st.warning("Please upload both job description and CV files.")
    
    elif page == "System Architecture":
        st.header("System Architecture")
        
        # Agent Architecture
        st.subheader("Agent Architecture")
        fig = plot_agent_architecture()
        st.pyplot(fig)
        
        # System Components
        st.subheader("System Components")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Core Components
            - **JD Summarizer**: Analyzes job descriptions
            - **CV Processor**: Processes and analyzes CVs
            - **Shortlisting Agent**: Matches candidates to requirements
            - **Scheduler**: Manages interview scheduling
            - **Coordinator**: Orchestrates the entire process
            """)
        
        with col2:
            st.markdown("""
            ### Technologies
            - **AgentNet**: Multi-agent coordination framework
            - **DeepSeek API**: Natural language processing
            - **Streamlit**: Interactive visualization
            - **PyPDF2**: PDF processing
            - **NetworkX**: Graph visualization
            """)
    
    elif page == "Results Analysis":
        st.header("Results Analysis")
        
        # Load results
        results = st.session_state.results or load_results()
        
        if results:
            try:
                # Metrics Overview
                st.subheader("Recruitment Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Candidates", len(results.get('candidates', [])))
                with col2:
                    st.metric("Shortlisted", len(results.get('shortlisting', {}).get('shortlisted', [])))
                with col3:
                    st.metric("Interviews Scheduled", len(results.get('scheduled_interviews', [])))
                
                # Visualizations
                st.subheader("Candidate Distribution")
                fig1 = plot_candidate_distribution(results)
                if fig1:
                    st.pyplot(fig1)
                else:
                    st.info("No candidate distribution data available.")
                
                st.subheader("Shortlisting Metrics")
                fig2 = plot_shortlisting_metrics(results)
                if fig2:
                    st.pyplot(fig2)
                else:
                    st.info("No shortlisting metrics data available.")
                
                # Detailed Results
                st.subheader("Detailed Results")
                if 'jd_summary' in results:
                    st.markdown("### Job Description Summary")
                    st.json(results['jd_summary'])
                
                if 'shortlisting' in results:
                    st.markdown("### Shortlisted Candidates")
                    st.json(results['shortlisting'])
            except Exception as e:
                st.error(f"Error displaying results: {str(e)}")
        else:
            st.info("No results available. Please process recruitment data first.")

if __name__ == "__main__":
    main()