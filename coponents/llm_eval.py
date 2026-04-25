from kfp import dsl
# from coponents.chunking import semantic_search
@dsl.component(
    base_image="python:3.12",
    packages_to_install=["langchain-nvidia-ai-endpoints>=1.2.1","python-dotenv>=1.2.2","pymilvus>=2.5.0","sentence-transformers>=2.2.0"]
)
def llm_eval(data:str)->str:
    import json
    import os
    import dotenv
    dotenv.load_dotenv()
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    client_eval=ChatNVIDIA(
        model=os.getenv("NVIDIA_MODEL2"),
        api_key=os.getenv("NVIDIA_API_KEY2"),
        top_p=1,
        temperature=0.3,
        max_completion_tokens=512,
    )
    def semantic_search(questoin:str,topK:int=2)->str:
        import os
        import json
        import dotenv
        dotenv.load_dotenv()
        from pymilvus import MilvusClient,DataType
        from sentence_transformers import SentenceTransformer
        client_milvus=MilvusClient(
            uri=os.getenv("MILVUS_URI"),
            token=os.getenv("MILVUS_TOKEN")
        )
        collection_name=os.getenv("MILVUS_COLLECTION_NAME")
        embedding_model=os.getenv("EMBEDDING_MODEL")
        embedd=SentenceTransformer(embedding_model)
        client_milvus.load_collection(collection_name)
        qv=embedd.encode([questoin],normalize_embeddings=True).tolist()[0]
        ans=client_milvus.search(
            collection_name=collection_name,
            data=[qv],
            limit=topK,
            output_fields=["chunk_id","text"],
            search_params={"metric_type":"COSINE"}
        )
        return json.dumps(ans,ensure_ascii=False)
    def first_json(text):
        text=text.strip().replace("```json","").replace("```","").strip()
        start_candidates=[i for i in [text.find("["),text.find("{")] if i!=-1]
        if not start_candidates:
            raise ValueError("No JSON found")
        start=min(start_candidates)
        obj,_=json.JSONDecoder().raw_decode(text[start:])
        return obj
    x=json.loads(data)
    if isinstance(x,str):
        x=json.loads(x)
    if isinstance(x,dict):
        x=[x]
    if not isinstance(x,list):
        raise ValueError("Format wrong guyssss")
    ans2=[]
    for i in x:
        c=i.get("question","")
        candi=i.get("candidate_answers",[])
        chunks=semantic_search(c,2)
        if isinstance(chunks,str):
            chunks=json.loads(chunks)

        # prompt=f"""
        # You are an expert evaluater. 
        # You have:
        # 1. Question: {c}
        # 2. Candidate answers: {candi}
        # 3. Relevant chunks: {chunks}
        # YOUR JOB IS TO UNDERSTAND THE QUESTION AND THE RELEVANT CHUNKS YOU HAVE. 
        # FROM THAT YOU HAVE TO EVALUATE EACH CANDIDATE ANSWER. 
        # 2. ACCEPT a candidate if it correctly answers the question AND its core facts align with the Ground Truth. Synonyms and rephrasing are perfectly fine.
        # 3. REJECT a candidate if it introduces new facts, industries, or concepts NOT found in the Ground Truth (e.g., hallucinating external knowledge).
        # 4. REJECT a candidate if it contradicts the Ground Truth or misses the core point.
        # 5. CRITICAL: You MUST include an evaluation object for EVERY single candidate answer provided in the input. Do NOT omit, skip, or filter out any candidate answer, regardless of whether it is accepted or rejected.
        # Output ONLY valid JSON without markdown code blocks (no ```json) in this exact format:
        # {{
        #     "question": "{c}",
        #     "candidate_answers": [
        #         {{
        #             "candidate_answer": "<insert candidate answer text here>",
        #             "status": "<accepted or rejected>"
        #         }}
        #     ]
        # }}
        # """
        prompt=f"""
        You are an expert evaluator.

        You have:
        1. Question: {c}
        2. Candidate answers: {json.dumps(candi,ensure_ascii=False)}
        3. Relevant chunks: {json.dumps(chunks,ensure_ascii=False)}

        Your job is to understand the question and the relevant chunks.
        From that, evaluate each candidate answer.

        1. ACCEPT a candidate if it correctly answers the question and its core facts align with the retrieved chunks.
        2. REJECT a candidate if it adds unsupported facts or goes beyond the retrieved chunks.
        3. REJECT a candidate if it contradicts the retrieved chunks or misses the core point.
        4. You must include an evaluation object for every candidate answer.

        Output ONLY valid JSON in this exact format:
        {{
            "question": "{c}",
            "evaluations": [
                {{
                    "candidate_answer": "<insert candidate answer text here>",
                    "status": "<accepted or rejected>"
                }}
            ]
        }}
        """
        eval_raw=client_eval.invoke(prompt).content
        try:
            p=first_json(eval_raw)
        except:
            p={
                "question":c,
                "evaluations":[
                    {"candidate_answer":x,"status":"rejected"}
                    for x in candi
                ]
            }
        if not isinstance(p,dict):
            p={
                "question":c,
                "evaluations":[
                    {"candidate_answer":x,"status":"rejected"}
                    for x in candi
                ]
            }

        if "question" not in p:
            p["question"]=c

        if "evaluations" not in p or not isinstance(p["evaluations"],list):
            p["evaluations"]=[
                {"candidate_answer":x,"status":"rejected"}
                for x in candi
            ]
        ans=[]
        sett=set()
        for j in p["evaluations"]:
            if isinstance(j,dict):
                cand=j.get("candidate_answer","").strip()
                status=j.get("status","rejected").strip().lower()
                if cand:
                    ans.append({
                        "candidate_answer":cand,
                        "status":"accepted" if status=="accepted" else "rejected"
                    })
                    sett.add(cand)

        for x in candi:
            if x not in sett:
                ans.append({
                    "candidate_answer":x,
                    "status":"rejected"
                })
        p["evaluations"]=ans
        ans2.append(p)
    return json.dumps(ans2,ensure_ascii=False)