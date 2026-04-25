import json
from kfp import dsl
@dsl.component(
    base_image="python:3.12",
    packages_to_install=["sqlalchemy>=2.0.25","psycopg2-binary>=2.9.12","python-dotenv>=1.2.2"]
)
def db_worker(data:str,stage:str)->str:
    import os
    from psycopg2 import connect
    import dotenv
    import json
    dotenv.load_dotenv()
    data=json.loads(data)
    if isinstance(data,dict):
        data=[data]
    for d in data:
        # i=d["question"]
        # j=d["original_ans"]
        # k=d["candidate_answers"]
        i=d.get("question","")
        j=d.get("original_ans","")
        # k=d.get("candidate_answers",[])
        conn=connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            dbname=os.getenv("DB_NAME"),
        )
        cur=conn.cursor()
        cur.execute("INSERT INTO storage2 (stage,question,original_answer,payload) VALUES (%s,%s,%s,%s)",(stage,i,j,json.dumps(d,ensure_ascii=False)))
        conn.commit()
        cur.close()
    conn.close()
    return "Data inserted successfully"