import os
import json
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise Exception("MONGO_URI not found in environment variables.")

client=MongoClient(MONGO_URI)
db=client['similarQ']
collection=db['question_metadata']

meta_path=os.path.abspath(os.path.join(os.path.dirname(__file__),"../../data/main_metadata.json"))
with open(meta_path,'r',encoding='utf-8') as f:
    metadata=json.load(f)

if isinstance(metadata,list):
    for q in metadata:
        collection.update_one(
            {"titleSlug":q['titleSlug']},
            {'$set':q},
            upsert=True
        )
else:
    collection.insert_one(metadata)
print("Upload complete")