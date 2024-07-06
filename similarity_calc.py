#IMPORTING
import csv
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

#Generating the embeddings
directory_cpp='LeetCode-Solutions\\C++' #check the drectory where you have downloaded the leetcode solutions
model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
i:int =0
complete_embeds=[]
question_names=[]
for file in os.listdir(directory_cpp):
    file_name=file.split('.')[0]
    file_path_cpp=os.path.join(directory_cpp,file)
    if os.path.isfile(file_path_cpp):
        with open(file_path_cpp,'r',encoding='UTF-8') as f:
            code=f.read()
        code_embedding=list(model.encode(code))
        name_embedding=list(model.encode(file_name))
        i+=1
        print(f'encoding done {i}')
        question_names.append(file_name)
        complete_embeds.append([file_name,name_embedding,code_embedding])

# Storing the Embeddings
with open('embeddings.csv','w',newline='') as f:
    w_obj=csv.writer(f)
    w_obj.writerows(complete_embeds)

#Reading Stored Embeddings as DataFrame
df=pd.read_csv('embeddings.csv')
df['Name_Embeddings']=df['Name_Embeddings'].apply(eval).apply(np.array)
names=[i for i in df['Name_Embeddings']]
df['Code_Embeddings']=df['Code_Embeddings'].apply(eval).apply(np.array)
codes=[i for i in df['Code_Embeddings']]

#Calculating Pairwise cosine similarity
names_similarity=[question_names,]+list(cosine_similarity(X=names,Y=names))
codes_similarity=[question_names,]+list(cosine_similarity(X=codes,Y=codes))

#Store Similarity Matrix
with open('names_similarity.csv','w') as nf:
    writer_ob=csv.writer(nf)
    writer_ob.writerows(names_similarity)
with open('codes_similarity.csv','w') as cf:
    writer_obj=csv.writer(cf)
    writer_obj.writerows(codes_similarity)

#Generating Combined Similarity Score
name_df=pd.read_csv('names_similarity.csv')
code_df=pd.read_csv('codes_similarity.csv')
comb_df=[[0,]+list(name_df.columns)]
for ind in name_df.index:
    row=[name_df.columns[ind],]
    for cols in name_df.columns:
        name_sim=round(name_df[cols][ind],4)
        code_sim=round(code_df[cols][ind],4)
        if code_sim<=0:
            code_sim=0
            name_sim=0
        if name_sim<0.6:
            name_sim=0
        if code_sim==0 and name_sim==0:
            row.append(0.0)
        else:
            row.append(round(((code_sim+name_sim)/((1 if name_sim!=0 else 0) + (1 if code_sim!=0 else 0))),4))
    comb_df.append(row)

#Storing Combined Similarity Scores
with open('combined_similarities.csv','w') as f:
    writer=csv.writer(f)
    writer.writerows(comb_df)