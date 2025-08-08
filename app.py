from flask import Flask, request, jsonify
from flask_cors import CORS
from search import sim_search,code_return,tags_return
import numpy as np

app= Flask(__name__)
CORS(app)

@app.route("/search",methods=['POST'])
def search():
    data=request.get_json()

    if not data or "name" not in data:
        return jsonify({"error":"Missing 'name' in request body"}),400
    
    data['name']=data['name'].replace(' ','-')

    result_np=sim_search(data['name'],code_return(data['name']),16)
    tags=tags_return(data['name'])
    result=[{
        "id":int(item['id']),
        "name":item['name'],
        "score":round(float(item['score']),4),
        "tags":tags
    } for item in result_np]
    return jsonify(result)

if __name__=='__main__':
    app.run(debug=True)