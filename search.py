import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import pandas as pd

tf=SentenceTransformer("all-MiniLM-L6-v2")
with open('question_map.json','r',encoding='utf-8') as qmap:
    mapping=json.load(qmap)

with open('question_tags.json','r',encoding='utf-8') as qmap:
    tags=json.load(qmap)

with open('question_index.json','r',encoding='utf-8') as qmap:
    indexes=json.load(qmap)
index_name=faiss.read_index('names.index')
index_code=faiss.read_index('codes.index')

def sim_search(q_title,q_code,k=16,alpha=0.5):
    name_embed=tf.encode([q_title],normalize_embeddings=True).astype('float32')
    code_embed=tf.encode([q_code],normalize_embeddings=True).astype('float32')

    name_scores,name_idx=index_name.search(name_embed,100)
    code_scores,code_idx=index_code.search(code_embed,100)

    from collections import defaultdict
    combined_scores = defaultdict(float)

    for idx, score in zip(name_idx[0], name_scores[0]):
        combined_scores[idx] += 0.70 * score

    for idx, score in zip(code_idx[0], code_scores[0]):
        combined_scores[idx] += 0.27 * score

    combined_scores_tuple = sorted(combined_scores.items(), key=lambda x: -x[1])[:30]
    c=0
    for idx,scores in combined_scores_tuple:
        if mapping[str(idx)]['name'] not in tags.keys():
            score=1
        else :
            s1=set(tags[mapping[str(idx)]['name']])
            s2=set(tags[q_title])
            score=len(s1.intersection(s2)) / len(s1.union(s2))
            if score==0 and c>k:
                break
            c+=1
        combined_scores[idx] += 0.03 * score
    top_combined = sorted(combined_scores.items(), key=lambda x: -x[1])[:k]
    # print(top_combined)
    result=[]
    for idx,score in top_combined:
        if mapping[str(idx)]['name']==q_title:
            continue
        result.append({
            'id':idx,
            'name':mapping[str(idx)]['name'],
            'score': round(score,4)
        })
    return result

def code_return(q_title):
    code=mapping[str(indexes[q_title])]['code']
    return code

def tags_return(q_title):
    return tags[q_title]