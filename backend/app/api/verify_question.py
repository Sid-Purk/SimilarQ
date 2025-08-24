from fastapi import APIRouter, Request
from pymongo import MongoClient
import re
from app.config import MONGO_URI


router=APIRouter()

client=MongoClient(MONGO_URI)
db=client['similarQ']
question_col=db['question_metadata']

def extract_slug(url: str):
    match= re.match(r"https://leetcode.com/problems/([^/]+)",url)
    return match.group(1) if match else None

def clean(doc):
    if not doc:
        return None
    doc = dict(doc)
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc
# print(extract_slug('https://leetcode.com/problems/two-sum/desc/ohib'))
@router.post("/api/verify-question")
async def verify_question(request: Request):
    data= await request.json()
    query= data.get("query", "").strip()
    if not query:
        return {"valid": False, "metadata":None}
    q={}
    if query.startswith("http"):
        slug=extract_slug(query)
        if slug:
            q=question_col.find_one({"titleSlug":slug})
            print('url')
            # return {"valid": bool(q), "metadata":clean(q)}
        else:
            return {"valid":False, "metadata":None}
    if bool(q):
        return {"valid": bool(q), "metadata":clean(q)}
    if re.match(r"[a-z0-2\-]+$", query):
        q=question_col.find_one({"titleSlug":query.lower()})
        print('slug')
        print(q)
        # return {"valid":bool(q), "metadata":clean(q)}
    if bool(q):
        return {"valid": bool(q), "metadata":clean(q)}

    l=[]
    for t in query.split(' '):
        if (t.startswith("I") or t.startswith("i")) and len(set(t))==1:
            l.append(t.upper())
        else:
            l.append(t.title())
    title=' '.join(l)
    q=question_col.find_one({"title":title})
    print(title)
    return {"valid":bool(q), "metadata":clean(q)}