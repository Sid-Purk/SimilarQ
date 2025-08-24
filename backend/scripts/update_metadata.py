import os
import json
from dotenv import load_dotenv
from pymongo import MongoClient
import requests
import time
import re

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
QUESTION_REPO_URL=os.getenv("QUESTION_REPO_URL")
if not MONGO_URI:
    raise Exception("MONGO_URI not found in environment variables.")
GITHUB_TOKEN=os.getenv("GITHUB_TOKEN")
REPO=os.getenv("REPO")
headers={
    "Authorization":f"token {GITHUB_TOKEN}",
}

client=MongoClient(MONGO_URI)
db=client['similarQ']
collection=db['question_metadata']

def get_mongo_sha():
    return {q['titleSlug']:q['github_sha'] for q in collection.find({},{'titleSlug':1, 'github_sha':1})}

def get_latest_commit():
    branch_resp=requests.get(f"https://api.github.com/repos/{REPO}/branches/master",headers=headers)
    return branch_resp.json()['commit']['sha']

def get_all_prog(commit_sha):
    tree_resp=requests.get(f"https://api.github.com/repos/{REPO}/git/trees/{commit_sha}?recursive=1",headers=headers)
    tree=tree_resp.json()['tree']

    cpp_files=[{
        'slug':os.path.splitext(os.path.basename(t['path']))[0],
        'github_sha':t['sha'],
        'github_path':t['path']
    } 
        for t in tree 
        if t['path'].startswith('C++/') and t['path'].endswith('.cpp')
    ]

    return cpp_files

def fetch_lc_metadata(slug):
    url="https://leetcode.com/graphql"
    query = """
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
    payload = {"query": query, "variables": {"titleSlug": slug}}
    resp = requests.post(url, json=payload)
    if resp.status_code == 200:
        q = resp.json()["data"]["question"]
        if q and "stats" in q and q["stats"]:
            try:
                stats = q["stats"]
                if isinstance(stats, str):
                    stats = json.loads(stats)
                q["totalAccepted"] = stats.get("totalAccepted")
                q["totalSubmission"] = stats.get("totalSubmission")
                q["totalAcceptedRaw"] = stats.get("totalAcceptedRaw")
                q["totalSubmissionRaw"] = stats.get("totalSubmissionRaw")
                q["acRate"] = stats.get("acRate")
            except Exception:
                pass
        q["titleSlug"] = slug
        q["url"] = f"https://leetcode.com/problems/{slug}/"
        return q
    else:
        print(f"LeetCode GraphQL error for {slug}: {resp.status_code}")
        return None

def raw_url(path):
    return f"https://raw.githubusercontent.com/{REPO}/master/C++/{path}.cpp"

patterns=[re.compile(r'^Weekly'),re.compile(r"^Biweekly")]

def remove(slug):
    return any(pattern.match(slug) for pattern in patterns)

def main():
    mongo_sha=get_mongo_sha()
    commit_sha=get_latest_commit()
    cpp_files=get_all_prog(commit_sha)
    
    for f in cpp_files:
        slug=f['titleSlug']
        sha=f['github_sha']

        new_question=slug not in mongo_sha.keys()
        if new_question or mongo_sha.get(slug)!=sha:
            code_resp=requests.get(raw_url(slug))
            code=code_resp.text
        else:
            code=None
        # check for weekly and bi-weekly tags
        meta=fetch_lc_metadata(slug)
        if not meta:
            print(f"Failed to fetch metadata for {slug}")
            continue

        update_doc = meta
        if code is not None:
            update_doc["code"] = code
        update_doc["github_sha"] = sha

        collection.update_one(
            {"titleSlug": slug},
            {"$set": update_doc},
            upsert=True
        )
        print(f"{'Inserted' if new_question else 'Updated'} {slug}")

        time.sleep(1)

def cleanup():
    for doc in collection.find({"tags":{"$exists":True,"$ne":[]}}, {"_id":1,"tags":1}):
        og_tags=doc['tags']
        cleaned=[tag for tag in og_tags if not remove(tag)]
        if len(cleaned)!=len(og_tags):
            collection.update_one(
                {"_id":doc["_id"]},
                {"$set":{"tags":cleaned}}
            )
            print(f"Cleaned tags for {doc['_id']}")

if __name__ == "__main__":
    main()