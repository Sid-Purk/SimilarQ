import requests
import os
import json
import time
import random

from dotenv import load_dotenv

# print(os.listdir('.\\lc_repo\\LeetCode-Solutions\\MySQL'))
load_dotenv()

GITHUB_TOKEN=os.getenv("GITHUB_TOKEN")
REPO=os.getenv("REPO")
headers={
    "Authorization":f"token {GITHUB_TOKEN}",
}

# corr=["wildcard-matching","permutations","dungeon-game","unique-binary-search-trees","balanced-binary-tree","populating-next-right-pointers-in-each-node-ii","populating-next-right-pointers-in-each-node","binary-tree-right-side-view","binary-tree-upside-down","binary-tree-right-side-view"]

url="https://leetcode.com/graphql"

cd=['last-person-to-fit-in-the-bus', 'classes-more-than-5-students', 'execute-cancellable-function-with-delay']

question_query="""
query questionData($titleSlug: String!){
    question(titleSlug: $titleSlug){
        questionId
        questionFrontendId
        title
        content
        likes
        dislikes
        stats
        similarQuestions
        categoryTitle
        hints
        topicTags { name }
        companyTags { name }
        difficulty
        isPaidOnly
        solution { canSeeDetail content }
        hasSolution
        hasVideoSolution
    }
}
"""

def fetch_question(title_slug):
    payload={
        "query":question_query,
        "variables":{
            "titleSlug":title_slug
        }
    }
    retries=8
    base_delay=2
    for tries in range(retries):
        try:
            response=requests.post(url,json=payload,timeout=15)
            delay=random.uniform(1,2)
            time.sleep(delay)
            if response.status_code==200:
                data=response.json()
                print(data)
                q=data['data']['question']
                q['url']=f"https://leetcode.com/problems/{title_slug}/"
                q['titleSlug']=title_slug
                stats = json.loads(q['stats'])
                q["totalAccepted"] = stats.get("totalAccepted")
                q["totalSubmission"] = stats.get("totalSubmission")
                q['totalAcceptedRaw']=stats.get("totalAcceptedRaw")
                q['totalSubmissionRaw']=stats.get('totalSubmissionRaw')
                q["acRate"] = stats.get("acRate")
                return q
            else:
                print(f"retrying {tries+1}/{retries} for {title_slug}")            
        except requests.exceptions.RequestException as e:
            print(f"Request error {title_slug}:{e}")
        backoff = base_delay * (2 ** tries) + random.uniform(0, 2)
        print(f"Retrying {tries + 1}/{retries} for {title_slug} after {backoff:.1f}s")
        time.sleep(backoff)
    print(f"Failed to fetch for {title_slug}")
    return None

new=[]
for c in cd:
    new.append(fetch_question(c))
print(new)
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
#         if (t['path'].startswith('C++/') and t['path'].endswith('.cpp')) or (t['path'].startswith('Python/') and os.path.splitext(os.path.basename(t['path']))[0] in corr) or (t['path'].startswith('TypeScript/') and t['path'].endswith('.ts')) or (t['path'].startswith('Shell/') and t['path'].endswith('.sh')) or (t['path'].startswith('MySQL/') and t['path'].endswith('.sql'))
#     } 

#     return cpp_files

# old_shas=get_all_prog(get_latest_commit())
# print(len(old_shas))
# with open('thanks.json','r',encoding='utf-8') as f:
#    correction=json.load(f)
# shas={}
# for s in old_shas.keys():
#   if s in correction.keys():
#     shas[correction[s]]=old_shas[s]
#   else:
#     shas[s]=old_shas[s]
# with open('question_shas.json','w',encoding='utf-8') as f:
#     json.dump(shas,f,ensure_ascii=False,indent=2)