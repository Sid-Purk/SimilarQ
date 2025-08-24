import os
import json

file=os.path.abspath(os.path.join(os.path.dirname(__file__),"../../data/main_metadata.json"))

with open(file,'r',encoding='utf-8') as f:
    data=json.load(f)

# print(data)
print(len(data))
