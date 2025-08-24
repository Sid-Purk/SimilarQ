import os
import requests
import json
import random
import time

url="https://leetcode.com/graphql"

allQ_query="""
query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput){
    problemsetQuestionList: questionList(
        categorySlug: $categorySlug
        limit: $limit
        skip: $skip
        filters: $filters
    ){
        total: totalNum
        questions: data{
            questionId
            questionFrontendId
            title
            titleSlug
        }
    }
}
"""

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

def fetch_all_questions():
    variables={
        "categorySlug":"",
        "limit":10000,
        "skip":0,
        "filters":{}
    }
    payload={
        "query":allQ_query,
        "variables":variables
    }
    response=requests.post(url,json=payload)
    if response.status_code!=200:
        print("Failed to fetch data")
        print(response.json())
        exit()
    data=response.json()
    questions=data['data']['problemsetQuestionList']['questions']
    print(f"Fetched {len(questions)} questions")
    return questions


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

def main():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Referer": "https://leetcode.com/problemset/all/"
    })

    questions = fetch_all_questions()
    all_questions=[]
    save_every=100

    dest=os.path.abspath(os.path.join(os.path.dirname(__file__),"../../data/question_metadata.json"))
    already_done=set()
    if os.path.exists(dest):
        with open(dest, "r", encoding="utf-8") as f:
            existing = json.load(f)
            all_questions = existing
            already_done = {q["questionFrontendId"] for q in all_questions}
        print(f"Resuming: {len(already_done)} questions already fetched")
    
    for i,q in enumerate(questions):
        title_slug=q['titleSlug']
        if q['questionFrontendId'] in already_done:
            continue
        print(f'{i}/{len(questions)} fetching details for {title_slug}')
        detail=fetch_question(title_slug)
        if detail:
            all_questions.append(detail)
        # Save progress every 100 questions
        if (len(all_questions) % save_every == 0) or (i == len(questions) - 1):
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(all_questions, f, indent=2, ensure_ascii=False)
            print(f"Saved progress ({len(all_questions)} questions)")
        # Every 100 requests, sleep longer (random 30-120 seconds)
        if (len(all_questions) % 100 == 0):
            long_pause = random.uniform(30, 120)
            print(f"Taking a longer break: {long_pause:.1f}s")
            time.sleep(long_pause)
    print(f"Saved all questions metadata")
    # with open(dest,"w",encoding="utf-8") as f:
    #     json.dump(all_questions,f,indent=2,ensure_ascii=False)
    # print(f"Saved all questions metadata")

if __name__=="__main__":
    main()