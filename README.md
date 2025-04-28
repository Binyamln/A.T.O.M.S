# ATOMS-GUI

## AI Powered Talent and Opportunity Matching System

ATOMS-GUI is a modern, AI-powered resume ranking system that helps match candidates with job opportunities based on multi-dimensional analysis of skills, experience, and job descriptions.

## Requirements

- **Python 3.10.11** (required)
- Virtual environment (recommended)
- Required Python packages (see requirements.txt)

C:\ATOMS>py -3.10 resume_ranking_gui.py


## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Binyamln/A.T.O.M.S
   cd ATOMS-GUI
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Install spaCy language model:
   ```
   python -m spacy download en_core_web_lg
   ```

## Usage

1. Run the application:
   ```
   python resume_ranking_gui.py
   ```

2. The application provides the following features:
   - Add candidates with their resumes
   - View and manage job descriptions
   - See rankings of candidates based on job requirements
   - View detailed matching scores

## File Structure

- `resume_ranking_gui.py` - Main application file
- `hybrid_matching_results.json` - Stores ranking results
- `requirements.txt` - Lists all required Python packages


## Important Notes

- The application uses the SentenceTransformer model 'all-mpnet-base-v2' for semantic matching
- Hybrid matching combines transformer embeddings with TF-IDF for improved accuracy
- Results are stored in the same directory as the script

## Troubleshooting

- If you encounter an error about missing spaCy model, run: `python -m spacy download en_core_web_lg`
- Ensure you're using Python 3.10.11 as specified in the requirements
- Make sure all dependencies are installed correctly