import os
import sys
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.agentnet_integration import RecruitingTask, AgentNetRecruitmentOrchestrator
from agentnet import AgentNet

# Initialize AgentNet with your API key
api_key = "your_api_key"  # Replace with your actual API key
agent_net = AgentNetRecruitmentOrchestrator(api_key=api_key)

# Sample job description for a Software Engineer position
jd_text = """
Senior Software Engineer

We are seeking a talented Senior Software Engineer to join our dynamic team. 
The ideal candidate will have:
- 5+ years of experience in Python development
- Strong background in machine learning and AI
- Experience with cloud platforms (AWS/Azure/GCP)
- Excellent problem-solving and communication skills

Responsibilities:
- Design and implement scalable software solutions
- Lead technical projects and mentor junior developers
- Collaborate with cross-functional teams
- Contribute to system architecture decisions
"""

# Sample candidate CVs
cv_texts = [
    """
John Smith
Senior Software Engineer

Experience:
- Lead Software Engineer at Tech Corp (6 years)
  - Led team of 5 developers on ML projects
  - Implemented cloud-native solutions on AWS
  - Python, TensorFlow, PyTorch expert

Education:
- M.S. Computer Science, Stanford University
- B.S. Computer Engineering, MIT
    """,
    """
Emily Johnson
Machine Learning Engineer

Experience:
- ML Engineer at AI Solutions (4 years)
  - Developed NLP models for text classification
  - Python, scikit-learn, deep learning
  - Azure and GCP certified

Education:
- Ph.D. Machine Learning, UC Berkeley
    """,
    """
Michael Chen
Software Developer

Experience:
- Software Developer at StartupX (3 years)
  - Full-stack development with Python
  - Basic ML model deployment
  - AWS Lambda and EC2 experience

Education:
- B.S. Computer Science, UCLA
    """
]

def print_section(title, content):
    """Helper function to print formatted sections"""
    print("\n" + "=" * 50)
    print(f"{title}:")
    print("-" * 50)
    if isinstance(content, dict):
        for key, value in content.items():
            print(f"{key}: {value}")
    elif isinstance(content, list):
        for item in content:
            print(f"- {item}")
    else:
        print(content)

# Run the full recruitment process
try:
    print("Starting recruitment process...\n")
    result = agent_net.run_full_process(jd_text, cv_texts)
    
    # Print JD Summary
    print_section("Job Description Summary", result["jd_summary"])
    
    # Print Processed Candidates
    print_section("Processed Candidates", result["candidates"])
    
    # Print Shortlisted Candidates
    print_section("Shortlisted Candidates", result["shortlisting"])
    
    # Print Interview Schedule
    print_section("Scheduled Interviews", result["scheduled_interviews"])
    
    # Print Performance Metrics
    print_section("AgentNet Performance Metrics", result["agentnet_metrics"])

except RuntimeError as e:
    print("\nError during recruitment process:")
    print(f"Error details: {str(e)}")

except Exception as e:
    print("\nUnexpected error occurred:")
    print(f"Error details: {str(e)}")