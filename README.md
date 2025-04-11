# ResuRank - AI-Powered Recruitment System

ResuRank is an advanced AI-powered recruitment system that automates the process of analyzing job descriptions, processing CVs, shortlisting candidates, and scheduling interviews using a multi-agent architecture.

## Features

- **Intelligent Job Description Analysis**: Extract key requirements and qualifications
- **Automated CV Processing**: Parse and analyze CVs in multiple formats (PDF, DOCX, TXT)
- **Smart Candidate Matching**: Match candidates to job requirements using AI
- **Automated Shortlisting**: Rank and shortlist candidates based on match scores
- **Interview Scheduling**: Automated interview scheduling with conflict resolution
- **Interactive Dashboard**: Visualize recruitment metrics and system architecture
- **Multi-Agent Architecture**: Coordinated agents for specialized tasks

## System Architecture

### Agent Architecture

The system uses a multi-agent architecture with specialized agents:

1. **JD Summarizer Agent**: Analyzes job descriptions and extracts key requirements
2. **CV Processor Agent**: Processes and analyzes CVs
3. **Shortlisting Agent**: Matches candidates to requirements and ranks them
4. **Scheduler Agent**: Manages interview scheduling
5. **Coordinator Agent**: Orchestrates the entire process

### Technology Stack

- **Backend**: Python, AgentNet Framework
- **AI/ML**: DeepSeek API, scikit-learn
- **Visualization**: Streamlit, Seaborn, NetworkX
- **Data Processing**: Pandas, NumPy
- **File Processing**: PyPDF2, python-docx

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/resurank.git
cd resurank
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export DEEPSEEK_API_KEY=your_api_key
```

## Usage

1. Start the Streamlit dashboard:
```bash
streamlit run app.py
```

2. Upload files:
   - Job Description (CSV format)
   - CVs (PDF, DOCX, or TXT format)

3. View results:
   - Recruitment metrics
   - Candidate distribution
   - Shortlisting results
   - System architecture

## Visualization Features

The system provides interactive visualizations:

1. **Agent Architecture Graph**: Visual representation of the multi-agent system
2. **Candidate Distribution**: Histogram of candidate match scores
3. **Shortlisting Metrics**: Pie chart of shortlisted vs. non-shortlisted candidates
4. **Recruitment Metrics**: Key performance indicators
5. **Detailed Results**: JSON view of processed data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- AgentNet Framework
- DeepSeek API
- Streamlit Community
- Open Source Community