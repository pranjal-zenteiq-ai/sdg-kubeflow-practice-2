from kfp import dsl
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
@dsl.component(
    base_image="python:3.12",
    packages_to_install=["langchain_nvidia_ai_endpoints>=1.2.1","python-dotenv>=1.2.2"],
)
def llm_seed_data(md_file:str)->str:
    import os
    import dotenv
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    dotenv.load_dotenv()
    print("model start")
    client_seed_data=ChatNVIDIA(
        model=os.getenv("NVIDIA_MODEL2"),
        api_key=os.getenv("NVIDIA_API_KEY2"),
        top_p=1,
        temperature=0.3,
        max_completion_tokens=256,
    )
    prompt=f"""You are given the following markdown content:
    {md_file}
    Task:
    Generate exactly 3 Question-Answer pairs based ONLY on the provided content.
    Rules:
    - Use only information explicitly present in the markdown.
    - Do not add, infer, or hallucinate any information.
    - Each question must be clear and specific.
    - Each answer must be complete and directly supported by the content.
    Output format (strict):
    [
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}}
    ]
    Return ONLY the JSON array.
    No extra text.
    Make me proud by following this perfectly.
    """
    seed_data=client_seed_data.invoke(prompt)
    print("model end")
    return seed_data.content.strip()
