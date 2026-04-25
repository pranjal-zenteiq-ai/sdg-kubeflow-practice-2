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
    # client_dean=ChatNVIDIA(
    #     model=os.getenv("NVIDIA_MODEL2"),
    #     api_key=os.getenv("NVIDIA_API_KEY2"),
    #     top_p=1,
    #     temperature=0.3,
    #     max_completion_tokens=256,
    # )
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
    ans=[]
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
        ans.append({
            "question":c,
            "original_ans":a,
            "candidate_answers":candidate_answers
        })
    return json.dumps(ans,ensure_ascii=False)