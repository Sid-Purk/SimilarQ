from fastapi import APIRouter, Request, HTTPException
from pinecone import Pinecone
import requests
from app.config import PINECONE_API_KEY,PINECONE_INDEX,HF_TOKEN, MODEL_SERVER_URL

router=APIRouter()

pc=Pinecone(api_key=PINECONE_API_KEY)
index=pc.Index(PINECONE_INDEX)

@router.post('/api/similar_search')
async def similar_search(request: Request):

    if not MODEL_SERVER_URL:
        raise HTTPException(status_code=500, detail="MODEL_SERVER_URL environment variable is not set.")
    
    data= await request.json()
    metadata=data.get('query','')
    
    combined=f"Name: {metadata['title']} | Code: {metadata['code']} | Tags: {','.join(metadata['tags'])}"

    embedding=None

    try:
        embed_url=f"{MODEL_SERVER_URL}/embed"

        response=requests.post(embed_url,json={"text":combined},timeout=90)
        response.raise_for_status()
        embedding=response.json()['embedding']
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Request to model server timed out (cold start may be in progress). Please try again in a moment.")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Model server is unavailable or returned an error: {e}")

    if not embedding:
        raise HTTPException(status_code=500, detail="Failed to get embedding from the model server.")

    res=index.query(vector=embedding,top_k=50, include_metadata=True)['matches']
    for meta in res:
        meta['metadata']['score']=meta['score']
    result=[match['metadata'] for match in res]

    return {'results': result}