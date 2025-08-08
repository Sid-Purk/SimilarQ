# SimilarQ: LeetCode Question Similarity Finder

## About

**SimilarQ** is a tool designed to help users find similar LeetCode questions for practice. It uses a natural language processing techniques, like embeddings and similarity search, to recommend related questions based on their titles, tags, and code. This project helps developers, improve their problem-solving skills by finding similar LeetCode questions. By exploring related problems, users can better understand different approaches and variations, making it easier to learn and practice coding concepts effectively.

The tool combines embeddings generated using `sentence-transformers` with FAISS for efficient similarity search and tag-based similarity scoring. It also includes an interactive frontend for easy access and a comprehensive dataset of LeetCode problems and solutions.

---

## Features Overview

- **Embedding Generation**: Uses `sentence-transformers` to generate embeddings for question names and code.
- **Similarity Search**: Implements FAISS for efficient similarity search across embeddings.
- **Tag-Based Similarity**: Computes similarity scores based on overlapping tags between questions.
- **Interactive Frontend**: A simple web interface to input a LeetCode question name or link and get similar questions.
- **Extensive Dataset**: Includes solutions and metadata for thousands of LeetCode problems.

---

## Installation

1. Clone the repository:
```bash
   git clone https://github.com/your-username/SimilarQ.git
   cd SimilarQ
```
2. Set up a virtual environment (optional but recommended):
```bash
    python -m venv venv
    source venv/bin/activate
```
3. Install the required dependencies:
```bash
    pip install -r requirements.txt
```
4. Run the backend application:
```bash
    python app.py
```
5. Access the frontend: Open your browser and navigate to the URL where the application is hosted (e.g., http://localhost:5000).

## Feedback
The development journey of this tool doesn't end here, and your input is crucial for its continuous improvement. Please feel free to report bugs or contribute by [submitting an issue](https://github.com/Sid-Purk/SimilarQ/issues).