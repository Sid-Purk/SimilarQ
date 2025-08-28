from fastapi import APIRouter, Request
from pinecone import Pinecone
from app.config import PINECONE_API_KEY,PINECONE_INDEX
# from app.services.model_service import get_model,model
from huggingface_hub import InferenceClient

router=APIRouter()

pc=Pinecone(api_key=PINECONE_API_KEY)
index=pc.Index(PINECONE_INDEX)

client=InferenceClient("Sid-the-sloth/leetcode_unixcoder_final")

@router.post('/api/similar_search')
async def similar_search(request: Request):
    data= await request.json()
    metadata=data.get('query','')
    
    combined=f"Name: {metadata['title']} | Code: {metadata['code']} | Tags: {','.join(metadata['tags'])}"

    # model=get_model()
    # embedding=model.encode(combined)
    embedding=client.feature_extraction(combined)

    res=index.query(vector=embedding.tolist(),top_k=50, include_metadata=True)['matches']
    for meta in res:
        meta['metadata']['score']=meta['score']
    result=[match['metadata'] for match in res]

    return {'results': result}