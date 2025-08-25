# SimilarQ: Find Similar LeetCode Questions

SimilarQ is a full-stack web application that helps users discover similar LeetCode questions using advanced code and text embeddings.  
It leverages MongoDB, Pinecone, and custom-trained transformer models for fast, accurate similarity search.

---

## Features

- 🔍 **Search for similar LeetCode questions** by name, slug, or URL
- 🏷️ **Filter results** by tags and difficulty
- 📊 **Similarity scoring** using fine-tuned transformer embeddings
- ⚡ **Fast, scalable backend** powered by FastAPI, MongoDB, and Pinecone
- 🎨 **Modern frontend** built with React, Vite, and TailwindCSS

---

## How It Works

1. **User enters a question** (name, slug, or URL).
2. **Backend verifies the question** using MongoDB metadata.
3. **Embeddings are generated** for the question using a fine-tuned UnixCoder model.
4. **Similarity search** is performed on Pinecone to find the top 50 similar questions.
5. **Results are filtered and displayed** with options to sort by tags and difficulty.

---

## Similarity Metric & Model Training

### Step 1: Baseline Grid Search

- **Embeddings for Name:** Generated using `Qwen3-embedding`.
- **Embeddings for Code:** Generated using `nomic-ai/nomic-embed-code`.
- **Tags Similarity:** Calculated using Jaccard similarity.
- **Grid Search:**
  - Used 20% of the dataset (positive pairs from LeetCode API, negative pairs from random selection).
  - Performed grid search to find optimal weights for name, code, and tag similarity.

### Step 2: Fine-Tuning UnixCoder

- **Data Preparation:**
  - Combined each question’s name, code, and tags into a single string.
  - Generated positive (similar) and negative (random) pairs.
- **Model Training:**
  - Fine-tuned the UnixCoder model on this dataset for similarity learning.
- **Embedding Generation:**
  - Used the trained UnixCoder model to generate embeddings for each question’s combined string.
  - Uploaded these embeddings to Pinecone for fast vector search.

### Step 3: Inference

- When a user enters a question:
  - The backend combines its name, code, and tags.
  - Generates an embedding using the fine-tuned UnixCoder model.
  - Performs a similarity search on Pinecone to return the top 50 most similar questions.

---

## Tech Stack

| Layer      | Technology                                                         |
| ---------- | ------------------------------------------------------------------ |
| Frontend   | React, Vite, TailwindCSS                                           |
| Backend    | FastAPI, Python, Uvicorn                                           |
| Database   | MongoDB Atlas                                                      |
| Vector DB  | Pinecone                                                           |
| Embeddings | Qwen3-embedding, nomic-ai/nomic-embed-code, UnixCoder (fine-tuned) |
| Deployment | Vercel (frontend), Render (backend)                                |

---

## Getting Started

### 1. Clone the Repo

```sh
git clone https://github.com/Sid-Purk/SimilarQ.git
cd SimilarQ
```

### 2. Backend Setup

- Install dependencies:
  ```sh
  pip install -r requirements.txt
  ```
- Set up your `.env` file with:
  ```
  MONGO_URI=your_mongodb_uri
  PINECONE_API_KEY=your_pinecone_api_key
  PINECONE_ENV=your_pinecone_env
  PINECONE_INDEX=your_pinecone_index
  MODEL_ID=your_google_drive_file_id
  ```
- Download the model from Google Drive before starting the backend:
  ```sh
  python backend/app/services/download_model.py
  ```
- Start the backend:
  ```sh
  uvicorn backend/app/main:app --host 0.0.0.0 --port 8000
  ```

### 3. Frontend Setup

- Go to the frontend folder:
  ```sh
  cd frontend
  ```
- Install dependencies:
  ```sh
  npm install
  ```
- Start the frontend:
  ```sh
  npm run dev
  ```
- Update API URLs in your frontend to point to your backend deployment.

---

## Deployment

- **Frontend:** Deploy on [Vercel](https://vercel.com/) (auto-detects Vite/React).
- **Backend:** Deploy on [Render](https://render.com/) (Python web service).
- **Model:** Store your trained model zip on Google Drive, download at backend startup using `gdown`.
- **Database:** Use [MongoDB Atlas](https://www.mongodb.com/atlas/database) for cloud-hosted metadata.
- **Vector DB:** Use [Pinecone](https://www.pinecone.io/) for similarity search.

---

## Usage

- Enter a LeetCode question name, slug, or URL.
- Filter results by tags or difficulty.
- Click on links for more details or solutions.

---

## Contributing

Pull requests and issues are welcome!  
If you can’t find a question, please [create an issue](https://github.com/Sid-Purk/SimilarQ/issues).

---

## Resources

- [My Github Repo](https://github.com/Sid-Purk/SimilarQ)
- [Kamyu's LeetCode Solutions](https://github.com/kamyu104/LeetCode-Solutions)

---

## License

MIT License

---

**Built by Sid Purkait | Powered by LeetCode, MongoDB, Pinecone, and Sentence Transformers**
