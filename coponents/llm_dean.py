from kfp import dsl

@dsl.component(
    base_image="python:3.12",
    packages_to_install=["langchain-nvidia-ai-endpoints>=1.2.1","python-dotenv>=1.2.2"],
)
def llm_dean(ans:str)->str:
    import os
    import dotenv
    from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA
    import json
    dotenv.load_dotenv()
    client_dean=ChatNVIDIA(
        model=os.getenv("NVIDIA_MODEL2"),
        api_key=os.getenv("NVIDIA_API_KEY2"),
        top_p=1,
        temperature=0.3,
        max_completion_tokens=512,
    )
    def first_json(text):
        text=text.strip().replace("```json","").replace("```","").strip()
        start_candidates=[i for i in [text.find("["),text.find("{")] if i!=-1]
        if not start_candidates:
            raise ValueError("No JSON found")
        start=min(start_candidates)
        obj,_=json.JSONDecoder().raw_decode(text[start:])
        return obj
    x=json.loads(ans)
    if isinstance(x,str):
        x=json.loads(x)
    if isinstance(x,dict):
        x=[x]
    if not isinstance(x,list):
        raise ValueError("Format wrong mate")
    dean_output=[]
    for i in x:
        c=i.get("question","")
        a=i.get("original_ans","")
        candidate_answers=i.get("candidate_answers",[])
        # prompt_dean=f"""
        # You are an expert evaluator.
        # Question: {c}
        # Ground Truth Answer: {a}
        # Candidate Answers to Evaluate: {json.dumps(candidate_answers,ensure_ascii=False)}
        # Evaluation Rules:
        # 1. Compare EACH candidate answer against the Ground Truth Answer.
        # 2. ACCEPT a candidate if it correctly answers the question AND its core facts align with the Ground Truth. Synonyms and rephrasing are perfectly fine.
        # 3. REJECT a candidate if it introduces new facts, industries, or concepts NOT found in the Ground Truth (e.g., hallucinating external knowledge).
        # 4. REJECT a candidate if it contradicts the Ground Truth or misses the core point.
        # Output ONLY valid JSON without markdown code blocks (no ```json) in this exact format:
        # {{
        # "question": "{c}",
        # "original_ans": "{a}",
        # "candidate_answers": ["<only accepted answers>"]
        # }}
        # """
        prompt_dean = f"""
        You are an expert evaluator.
        Question: {c}
        Ground Truth Answer: {a}
        Candidate Answers to Evaluate: {json.dumps(candidate_answers, ensure_ascii=False)}
        Evaluation Rules:
        1. Compare EACH candidate answer against the Ground Truth Answer.
        2. ACCEPT a candidate if it correctly answers the question AND its core facts align with the Ground Truth. Synonyms and rephrasing are perfectly fine.
        3. REJECT a candidate if it introduces new facts, industries, or concepts NOT found in the Ground Truth (e.g., hallucinating external knowledge).
        4. REJECT a candidate if it contradicts the Ground Truth or misses the core point.
        5. CRITICAL: You MUST include an evaluation object for EVERY single candidate answer provided in the input. Do NOT omit, skip, or filter out any candidate answer, regardless of whether it is accepted or rejected.
        Output ONLY valid JSON without markdown code blocks (no ```json) in this exact format:
        {{
            "question": "{c}",
            "original_ans": "{a}",
            "evaluations": [
                {{
                    "candidate_answer": "<insert candidate answer text here>",
                    "status": "<accepted or rejected>"
                }}
            ]
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
        # dean_raw=client_dean.invoke(prompt_dean).content
        # try:
        #     p=first_json(dean_raw)
        # except:
        #     p={"question":c,"original_ans":a,"candidate_answers":candidate_answers}

        # if not isinstance(p,dict):
        #     p={"question":c,"original_ans":a,"candidate_answers":candidate_answers}

        # if "question" not in p:
        #     p["question"]=c
        # if "original_ans" not in p:
        #     p["original_ans"]=a
        # if "candidate_answers" not in p or not isinstance(p["candidate_answers"],list):
        #     p["candidate_answers"]=candidate_answers

        # dean_output.append(p)
        dean_raw=client_dean.invoke(prompt_dean).content
        try:
            p=first_json(dean_raw)
        except:
            p={
                "question":c,
                "original_ans":a,
                "evaluations":[
                    {"candidate_answer":x,"status":"rejected"}
                    for x in candidate_answers
                ]
            }
        if not isinstance(p,dict):
            p={
                "question":c,
                "original_ans":a,
                "evaluations":[
                    {"candidate_answer":x,"status":"rejected"}
                    for x in candidate_answers
                ]
            }
        if "question" not in p:
            p["question"]=c
        if "original_ans" not in p:
            p["original_ans"]=a

        if "evaluations" not in p or not isinstance(p["evaluations"],list):
            p["evaluations"]=[
                {"candidate_answer":x,"status":"rejected"}
                for x in candidate_answers
            ]
        clean_evals=[]
        seen=set()
        for j in p["evaluations"]:
            if isinstance(j,dict):
                cand=j.get("candidate_answer","").strip()
                status=j.get("status","rejected").strip().lower()
                if cand:
                    clean_evals.append({
                        "candidate_answer":cand,
                        "status":"accepted" if status=="accepted" else "rejected"
                    })
                    seen.add(cand)
        for x in candidate_answers:
            if x not in seen:
                clean_evals.append({
                    "candidate_answer":x,
                    "status":"rejected"
                })
        p["evaluations"]=clean_evals
        dean_output.append(p)
    return json.dumps(dean_output)