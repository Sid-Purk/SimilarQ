import json
import pandas as pd
import pickle
import os
import requests
from dotenv import load_dotenv

# print(os.listdir('.\\lc_repo\\LeetCode-Solutions\\MySQL'))
load_dotenv()

GITHUB_TOKEN=os.getenv("GITHUB_TOKEN")
REPO=os.getenv("REPO")
headers={
    "Authorization":f"token {GITHUB_TOKEN}",
}
# corr=["wildcard-matching","permutations","dungeon-game","unique-binary-search-trees","balanced-binary-tree","populating-next-right-pointers-in-each-node-ii","populating-next-right-pointers-in-each-node","binary-tree-right-side-view","binary-tree-upside-down","binary-tree-right-side-view"]
# not_there=["last-person-to-fit-in-the-bus",'jump-game-viii']
# code={}

# for c in os.listdir('.\\lc_repo\\LeetCode-Solutions\\MySQL'):
#   print(c.split('.')[0])
#   with open(('.\\lc_repo\\LeetCode-Solutions\\MySQL\\'+c),'r') as f:
#     pycode=f.read()
#   code[c.split('.')[0]]=pycode

# with open('main_metadata2.json','r',encoding='utf-8') as f:
#   metacode = json.load(f)
# print(len(metacode))
# for key in code:
#   metacode.append({
#     'slug':key,
#     'code':code[key]
#   })

# print(len(metacode))
# with open('main_metadata2.json','w',encoding='utf-8') as f:
#   json.dump(metacode,f,ensure_ascii=False,indent=2)
# ---------------------------------------------------------------------------
# with open('thanks.json','r',encoding='utf-8') as f:
#   correction = json.load(f)

# with open('code_embeds.pkl','rb') as f:
#   nembeds=pickle.load(f)

# c=0
# for q in correction.keys():
#   if q in nembeds.keys():
#     c+=1
# print(c)


with open('code_metadata.json','r',encoding='utf-8') as f:
  code = json.load(f)
print(len(code)) # 3624
with open('question_metadata.json','r',encoding='utf-8') as f:
  question = json.load(f)
print(len(question)) #3646
with open('question_shas.json','r',encoding='utf-8') as f:
  shas = json.load(f)
print(len(shas)) #3646
with open('main_metadata.json','r',encoding='utf-8') as f:
  main = json.load(f)
print(len(main)) # 3183
c={}
for x in code:
  c[x['slug']]=x['code']

q={}
for x in question:
  q[x['titleSlug']]=x

m={}
for meta in main:
  m[meta['titleSlug']]=meta


# def get_latest_commit():
#     branch_resp=requests.get(f"https://api.github.com/repos/{REPO}/branches/master",headers=headers)
#     return branch_resp.json()['commit']['sha']

# def get_all_prog(commit_sha):
#     tree_resp=requests.get(f"https://api.github.com/repos/{REPO}/git/trees/{commit_sha}?recursive=1",headers=headers)
#     tree=tree_resp.json()['tree']

#     cpp_files={
#         os.path.splitext(os.path.basename(t['path']))[0]:t['sha']
#         # 'github_path':t['path']
#         for t in tree 
#         if t['path'].startswith('C++/') and t['path'].endswith('.cpp')
#     } 

#     return cpp_files

# old_shas=get_all_prog(get_latest_commit())
# shas={}
# for s in old_shas.keys():
#   if s in correction.keys():
#     shas[correction[s]]=old_shas[s]
#   else:
#     shas[s]=old_shas[s]

print(len(shas))
new=[]
for q in question:
  if q['titleSlug'] in m.keys():
    new.append({
      'titleSlug':q['titleSlug'],
      'title':q['title'],
      'questionId':q['questionId'],
      'content':q['content'],
      'difficulty':q['difficulty'],
      'categoryTitle':q['categoryTitle'],
      'isPaidOnly':q['isPaidOnly'],
      'similarQuestions':[t['titleSlug'] for t in json.loads(q['similarQuestions'])],
      'tags':[t['name'] for t in q['topicTags']],
      "code": c[q['titleSlug']],
      "url": q['url'],
      "totalAccepted": q['totalAccepted'],
      "totalSubmission": q["totalSubmission"],
      "totalAcceptedRaw": q["totalAcceptedRaw"],
      "totalSubmissionRaw": q["totalSubmissionRaw"],
      "acRate": q['acRate'],
      "github_sha": shas[q['titleSlug']]
    })
  elif q['titleSlug'] in c.keys():
    new.append({
      'titleSlug':q['titleSlug'],
      'title':q['title'],
      'questionId':q['questionId'],
      'content':q['content'],
      'difficulty':q['difficulty'],
      'categoryTitle':q['categoryTitle'],
      'isPaidOnly':q['isPaidOnly'],
      'similarQuestions':[t['titleSlug'] for t in json.loads(q['similarQuestions'])],
      'tags':[t['name'] for t in q['topicTags']],
      "code": c[q['titleSlug']],
      "url": q['url'],
      "totalAccepted": q['totalAccepted'],
      "totalSubmission": q["totalSubmission"],
      "totalAcceptedRaw": q["totalAcceptedRaw"],
      "totalSubmissionRaw": q["totalSubmissionRaw"],
      "acRate": q['acRate'],
      "github_sha": shas[q['titleSlug']]
    })
print(len(new))
with open("main_metadata2.json",'w',encoding='utf-8') as f:
  json.dump(new,f,ensure_ascii=False,indent=2)
# main_slug=set([x['titleSlug'] for x in main])
# code_slug=set([x['slug'] for x in code])
# print(code_slug-main_slug)

# new_code=[]
# for x in code:
#   if x['slug'] in correction.keys():
#     if correction[x['slug']]=='delete':
#       print('avoided: ',x['slug'])
#       continue
#     print('updating: ', x['slug'])
#     new_code.append({
#       'slug':correction[x['slug']],
#       'code':x['code']
#     })
#   else:
#     new_code.append(x)
    
# with open('code_metadata.json','w',encoding='utf-8') as f:
#   json.dump(new_code,f,ensure_ascii=False,indent=2)
# q=set([x['titleSlug'] for x in question if '2' in x['titleSlug']])
# print(q)
# print()
# print('codes extra:')
# diff1=p-q
# print(diff1)
# print()
# print('leetcode extra:')
# diff2=q-p
# print(diff2)
# q=[x['slug'] for x in code if (x['slug'])]
# p=[]
# for x in code:
#   print(x['slug'])
#   ch=input("yes? : ")
#   if ch=='y' or ch=='yes':
#     p.append(x['slug'])

# with open('thanks.txt','w') as f:
#   for x in p:
#     f.write(f'{x}\n')
# print(p)