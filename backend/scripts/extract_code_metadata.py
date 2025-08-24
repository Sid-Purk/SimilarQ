import os
import json

def extract_cpp_code(repo_dir,dest):
    cpp_dir=os.path.join(repo_dir,"C++")
    metadata=[]
    for f in os.listdir(cpp_dir):
        if f.endswith(".cpp"):
            slug=f.split('.')[0]
            with open(os.path.join(cpp_dir,f), encoding="utf-8", errors="ignore") as fname:
                code=fname.read()
            metadata.append({
                "slug":slug,
                "code":code
            })
    with open(dest,'w',encoding="utf-8") as f:
        json.dump(metadata,f,ensure_ascii=False,indent=2)
    print(f"Extracted {len(metadata)} C++ solutions to {dest}")

if __name__=="__main__":
    repo_dir=os.path.abspath(os.path.join(os.path.dirname(__file__),"../../data/lc_repo/LeetCode-Solutions"))
    dest=os.path.abspath(os.path.join(os.path.dirname(__file__),"../../data/code_metadata.json"))
    extract_cpp_code(repo_dir,dest)