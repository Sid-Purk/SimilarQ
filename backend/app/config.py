import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI=os.getenv("MONGO_URI")
MONGODB_USER_PASSWORD=os.getenv("MONGODB_USER_PASSWORD")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GITHUB_TOKEN=os.getenv("GITHUB_TOKEN")
MODEL_PATH=os.getenv("MODEL_PATH",'leetcode_unixcoder_final')
PINECONE_INDEX=os.getenv("PINECONE_INDEX",'similarq')
PINECONE_ENV=os.getenv("PINECONE_ENV")
MODEL_ID=os.getenv("MODEL_ID")