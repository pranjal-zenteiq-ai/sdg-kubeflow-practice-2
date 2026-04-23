from kfp import dsl

@dsl.component(
    base_image="python:3.12",
    packages_to_install=["langchain_nvidia_ai_endpoints>=1.2.1","python-dotenv>=1.2.2"],
)
def llm_candidate_generator(seed_data:str)->str:
    import os
    import json
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    # seed_data_dic=json.loads(seed_data)
    import dotenv
    dotenv.load_dotenv()
    client_candidate_generator=ChatNVIDIA(
        model=os.getenv("NVIDIA_MODEL2"),
        api_key=os.getenv("NVIDIA_API_KEY2"),
        top_p=1,
        temperature=0.3,
        max_completion_tokens=256,
    )
    client_dean=ChatNVIDIA(
        model=os.getenv("NVIDIA_MODEL2"),
        api_key=os.getenv("NVIDIA_API_KEY2"),
        top_p=1,
        temperature=0.3,
        max_completion_tokens=256,
    )
    def first_json(text):
        text=text.strip().replace("```json","").replace("```","").strip()
        start_candidates=[i for i in [text.find("["),text.find("{")] if i!=-1]
        if not start_candidates:
            raise ValueError("No JSON found")
        start=min(start_candidates)
        obj,_=json.JSONDecoder().raw_decode(text[start:])
        return obj
    x=json.loads(seed_data)
    if isinstance(x,str):
        x=json.loads(x)
    if isinstance(x,dict):
        x=[x]
    if not isinstance(x,list):
        raise ValueError("Invalid input format, list olyyyyy")
    dean_output=[]
    for i in x:
        c=i.get("question","")
        a=i.get("answer","")
        prompt=f"""
        You are an expert AI data generator. I will give you a Question. 
        Your task is to generate 3 distinct, accurate, and meaningful answers to this question.
        Question: {c}
        Rules:
        1. The 3 answers must be factually correct and directly answer the question.
        2. Vary the length and sentence structure.
        3. Output ONLY valid JSON in this exact format, with no markdown code blocks (no ```json):
        [
        {{"answer": "..."}},
        {{"answer": "..."}},
        {{"answer": "..."}}
        ]
        """
        candi_raw=client_candidate_generator.invoke(prompt).content
        try:
            candi_json=first_json(candi_raw)
        except:
            candi_json=[{"answer":a}]
        if isinstance(candi_json,dict):
            candi_json=[candi_json]
        candidate_answers=[]
        for j in candi_json:
            if isinstance(j,dict) and j.get("answer"):
                candidate_answers.append(j["answer"].strip())
        prompt_dean=f"""
        You are an expert evaluator.
        Question: {c}
        Ground Truth Answer: {a}
        Candidate Answers to Evaluate: {json.dumps(candidate_answers,ensure_ascii=False)}
        Evaluation Rules:
        1. Compare EACH candidate answer against the Ground Truth Answer.
        2. ACCEPT a candidate if it correctly answers the question AND its core facts align with the Ground Truth. Synonyms and rephrasing are perfectly fine.
        3. REJECT a candidate if it introduces new facts, industries, or concepts NOT found in the Ground Truth (e.g., hallucinating external knowledge).
        4. REJECT a candidate if it contradicts the Ground Truth or misses the core point.
        Output ONLY valid JSON without markdown code blocks (no ```json) in this exact format:
        {{
        "question": "{c}",
        "original_ans": "{a}",
        "candidate_answers": ["<only accepted answers>"]
        }}
        """
        # try:
        #     p=json.loads(client_dean.invoke(prompt_dean).content)
        # except:
        #     p={"question":c,"original_ans":i.get("answer"),"candidate_answers":[candi_ans.content]}
        # try:
        #     p=json.loads(client_dean.invoke(prompt_dean))
        # except:
        #     p={"question":c,"original_ans":i.get("answer"),"candidate_answers":[candi_ans.content]}
        dean_raw=client_dean.invoke(prompt_dean).content
        try:
            p=first_json(dean_raw)
        except:
            p={"question":c,"original_ans":a,"candidate_answers":candidate_answers}

        if not isinstance(p,dict):
            p={"question":c,"original_ans":a,"candidate_answers":candidate_answers}

        if "question" not in p:
            p["question"]=c
        if "original_ans" not in p:
            p["original_ans"]=a
        if "candidate_answers" not in p or not isinstance(p["candidate_answers"],list):
            p["candidate_answers"]=candidate_answers

        dean_output.append(p)  
    return json.dumps(dean_output)