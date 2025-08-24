import os
import json
import requests
import re

GITHUB_TOKEN=os.getenv("GITHUB_TOKEN")
REPO=os.getenv("REPO")
headers={
    "Authorization":f"token {GITHUB_TOKEN}",
}

branch_resp=requests.get(f"https://api.github.com/repos/{REPO}/branches/master",headers=headers)
commit_sha=branch_resp.json()['commit']['sha']

tree_resp=requests.get(f"https://api.github.com/repos/{REPO}/git/trees/{commit_sha}?recursive=1",headers=headers)
tree=tree_resp.json()['tree']

cpp_files=[t for t in tree if t['path'].startswith('C++/') and t['path'].endswith('.cpp')]

slug_to_sha={os.path.splitext(os.path.basename(f['path']))[0]:f['sha'] for f in cpp_files}

patterns=[re.compile(r'^Weekly'),re.compile(r"^Biweekly")]

def remove(slug):
    return any(pattern.match(slug) for pattern in patterns)

def combine_metadata(code_path,details_path,dest):
    with open(code_path,"r",encoding="utf-8") as f:
        code_data={q["slug"]:q["code"] for q in json.load(f)}
    with open(details_path,"r",encoding="utf-8") as f:
        details_data={q["titleSlug"]:q for q in json.load(f)}
    combined_metadata=[]
    for slug,code in code_data.items():
        meta=details_data.get(slug)
        if not meta:
            continue
        combined_metadata.append({
            "titleSlug":slug,
            "title":meta['title'],
            "questionId":meta["questionId"],
            "content":meta["content"],
            "difficulty":meta['difficulty'],
            "categoryTitle":meta["categoryTitle"],
            "isPaidOnly":meta["isPaidOnly"],
            "tags":[t['name'] for t in meta['topicTags'] if not remove(t['name'])],
            "code":code,
            "url":meta['url'],
            "totalAccepted":meta['totalAccepted'],
            "totalSubmission":meta['totalSubmission'],
            "totalAcceptedRaw":meta['totalAcceptedRaw'],
            "totalSubmissionRaw":meta['totalSubmissionRaw'],
            'acRate':meta['acRate'],
            "github_sha":slug_to_sha[slug]
        })
    with open(dest,"w",encoding="utf-8") as f:
        json.dump(combined_metadata,f,ensure_ascii=False,indent=2)
    print(f"Completed metadata of {len(combined_metadata)}")

if __name__=="__main__":
    code_path=os.path.abspath(os.path.join(os.path.dirname(__file__),"../../data/code_metadata.json"))
    details_path=os.path.abspath(os.path.join(os.path.dirname(__file__),"../../data/question_metadata.json"))
    dest=os.path.abspath(os.path.join(os.path.dirname(__file__),"../../data/main_metadata.json"))
    combine_metadata(code_path,details_path,dest)