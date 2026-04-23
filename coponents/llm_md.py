from langchain_nvidia_ai_endpoints import ChatNVIDIA
from kfp import dsl 
import os
@dsl.component(
    base_image="python:3.12",
    packages_to_install=["langchain-nvidia-ai-endpoints>=1.2.1","pypdf>=6.10.2","minio>=7.2.20","python-dotenv>=1.2.2"],
)
def llm_md(path:str)->str:
    import os
    from pypdf import PdfReader
    from minio import Minio
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    import dotenv
    dotenv.load_dotenv()
    print("Minio start")
    client_minio=Minio(
        endpoint=os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=False,
    )
    print("Minio end")
    local="/tmp/mypdf.pdf"  
    client_minio.fget_object("pdf","stem_topicccc.pdf",local)
    print("Minio end2")
    x=PdfReader(local)
    text=""
    op =[]
    client_md=ChatNVIDIA(
    model=os.getenv("NVIDIA_MODEL2"),
    api_key=os.getenv("NVIDIA_API_KEY2"),
    temperature=0.1,
    top_p=1,
    max_completion_tokens=256,
)
    op=[]

    for i in x.pages:
        chunk=i.extract_text()
        if not chunk:
            continue
        prompt=f"""
            You are a expert in pdf to markdown conversion. You are given an input {chunk},
            and you have to convert it into proper correct markdown format making sure that :
            - headings,titles,sub-titles(#,##,###) are all semantically correct.
            - NO EXTRA IS ADDED OR REMOVED FROM THE TEXT PROVIDED TO YOU
            - YOU ONLY HAVE TO TRANSFORM THE SCHEMA TO MARKDOWN FORMAT
            Make me proud by doing all this without mistake and proper break statements and provide a output as a proper markdown file.
        """

        md=client_md.invoke(prompt)
        op.append(md.content)
    return "\n".join(op)