from pinecone import Pinecone
from dotenv import load_dotenv
import os
import json
from sentence_transformers import SentenceTransformer
import time

start_time=time.perf_counter()

load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")

pc=Pinecone(api_key=PINECONE_API_KEY)
index=pc.Index('similarq')
print(index.describe_index_stats())

model=SentenceTransformer('../leetcode_unixcoder_final')
print(model)

with open(os.path.abspath(os.path.join(os.path.dirname(__file__),'../../data/main_metadata.json')),'r',encoding='utf-8') as f:
    metadata=json.load(f)

vector_upload=[]
i=1
for meta in metadata:
    q = f"Name: {meta['title']} | Code: {meta['code']} | Tags: {','.join(meta['tags'])}"
    embed=model.encode(q)
    print(f"{i}: {meta['title']}")
    vector_upload.append({
        'id': meta['titleSlug'],
        'values': embed,
        'metadata':{
            'problem_id':meta['questionId'],
            'name':meta['title'],
            'tags':meta['tags'],
            'difficulty':meta['difficulty'],
            'url':meta['url'],
            'acRate':meta['acRate'],
            'isPaid':meta['isPaidOnly'],
        }
    })
    i+=1

print(len(vector_upload))
# index.upsert(vectors=vector_upload)  # too large for one upload

batch_sz=50
for i in range(0,len(vector_upload),batch_sz):
    batch=vector_upload[i:i+batch_sz]
    print(f"Uploading batch {i//batch_sz + 1}/{(len(vector_upload) + batch_sz - 1)//batch_sz}")
    try:
        index.upsert(vectors=batch)
        print(f"Uploaded {i//batch_sz +1}")
    except Exception as e:
        print("Trying again")
        time.sleep(1)
        index.upsert(vectors=batch)
    time.sleep(0.1)

print(index.describe_index_stats())
end_time=time.perf_counter()
print(start_time-end_time)