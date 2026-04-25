from kfp import dsl,compiler
import dotenv
dotenv.load_dotenv()
from coponents.llm_md import llm_md
from coponents.llm_seed_data import llm_seed_data
from coponents.llm_candidate_generator import llm_candidate_generator
from coponents.db_worker import db_worker
from coponents.llm_eval import llm_eval
from coponents.llm_dean import llm_dean
from coponents.chunking import chunking

@dsl.pipeline(
    name="daddy-pipeline",
    description="My daddy pipeline"
)
def daddy_pipeline(path:str):
    import os
    import dotenv
    dotenv.load_dotenv()
    md=llm_md(path=path)
    # llm_md.set_cache_options(enable_cache=False)
    md.set_caching_options(enable_caching=False)
    md.set_env_variable("MINIO_ENDPOINT", os.getenv("MINIO_ENDPOINT"))
    md.set_env_variable("MINIO_ACCESS_KEY", os.getenv("MINIO_ACCESS_KEY"))
    md.set_env_variable("MINIO_SECRET_KEY", os.getenv("MINIO_SECRET_KEY"))
    md.set_env_variable("NVIDIA_API_KEY2", os.getenv("NVIDIA_API_KEY2"))
    md.set_env_variable("NVIDIA_MODEL2", os.getenv("NVIDIA_MODEL2"))
    seed_data=llm_seed_data(md_file=md.output)
    # llm_seed_data.set_cache_options(enable_cache=False)
    seed_data.set_caching_options(enable_caching=False)
    seed_data.set_env_variable("NVIDIA_API_KEY2", os.getenv("NVIDIA_API_KEY2"))
    seed_data.set_env_variable("NVIDIA_MODEL2", os.getenv("NVIDIA_MODEL2"))
    candidate_answers=llm_candidate_generator(seed_data=seed_data.output)
    # llm_candidate_generator.set_cache_options(enable_cache=False)
    candidate_answers.set_caching_options(enable_caching=False)
    candidate_answers.set_env_variable("NVIDIA_API_KEY2", os.getenv("NVIDIA_API_KEY2"))
    candidate_answers.set_env_variable("NVIDIA_MODEL2", os.getenv("NVIDIA_MODEL2"))
    dean=llm_dean(ans=candidate_answers.output)
    dean.set_caching_options(enable_caching=False)
    dean.set_env_variable("NVIDIA_API_KEY2", os.getenv("NVIDIA_API_KEY2"))
    dean.set_env_variable("NVIDIA_MODEL2", os.getenv("NVIDIA_MODEL2"))
    chunk=chunking(md=md.output)
    chunk.set_caching_options(enable_caching=False)
    chunk.set_env_variable("MILVUS_URI", os.getenv("MILVUS_URI"))
    chunk.set_env_variable("MILVUS_TOKEN", os.getenv("MILVUS_TOKEN"))
    chunk.set_env_variable("MILVUS_COLLECTION_NAME", os.getenv("MILVUS_COLLECTION_NAME"))
    chunk.set_env_variable("EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL"))
    eval=llm_eval(data=candidate_answers.output)
    eval.set_env_variable("NVIDIA_API_KEY2", os.getenv("NVIDIA_API_KEY2"))
    eval.set_env_variable("NVIDIA_MODEL2", os.getenv("NVIDIA_MODEL2"))
    eval.set_env_variable("MILVUS_URI", os.getenv("MILVUS_URI"))
    eval.set_env_variable("MILVUS_TOKEN", os.getenv("MILVUS_TOKEN"))
    eval.set_env_variable("MILVUS_COLLECTION_NAME", os.getenv("MILVUS_COLLECTION_NAME"))
    eval.set_env_variable("EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL"))
    eval.set_caching_options(enable_caching=False)
    eval.after(chunk)
    # c=db_worker(data=candidate_answers.output)
    # c.set_caching_options(enable_caching=False)
    # c.set_env_variable("DB_HOST", os.getenv("DB_HOST"))
    # c.set_env_variable("DB_USER", os.getenv("DB_USER"))
    # c.set_env_variable("DB_PASSWORD", os.getenv("DB_PASSWORD"))
    # c.set_env_variable("DB_NAME", os.getenv("DB_NAME"))
    c2=db_worker(data=dean.output,stage="dean")
    c2.set_caching_options(enable_caching=False)
    c2.set_env_variable("DB_HOST", os.getenv("DB_HOST"))
    c2.set_env_variable("DB_USER", os.getenv("DB_USER"))
    c2.set_env_variable("DB_PASSWORD", os.getenv("DB_PASSWORD"))
    c2.set_env_variable("DB_NAME", os.getenv("DB_NAME"))
    c3=db_worker(data=eval.output,stage="eval")
    c3.set_caching_options(enable_caching=False)
    c3.set_env_variable("DB_HOST", os.getenv("DB_HOST"))
    c3.set_env_variable("DB_USER", os.getenv("DB_USER"))
    c3.set_env_variable("DB_PASSWORD", os.getenv("DB_PASSWORD"))
    c3.set_env_variable("DB_NAME", os.getenv("DB_NAME"))
    print("Pipeline completed successfully")
    print("MINIO_ENDPOINT:", os.getenv("MINIO_ENDPOINT"))
    print("MINIO_ACCESS_KEY:", os.getenv("MINIO_ACCESS_KEY"))
    print("MINIO_SECRET_KEY:", os.getenv("MINIO_SECRET_KEY"))
    print("NVIDIA_MODEL2",os.getenv("NVIDIA_MODEL2"))
    print("NVIDIA_API_KEY2",os.getenv("NVIDIA_API_KEY2"))
    print("DB_HOST",os.getenv("DB_HOST"))
    print("DB_USER",os.getenv("DB_USER"))
    print("DB_PASSWORD",os.getenv("DB_PASSWORD"))
    print("DB_NAME",os.getenv("DB_NAME"))
    print("MILVUS_URI",os.getenv("MILVUS_URI"))
    print("MILVUS_TOKEN",os.getenv("MILVUS_TOKEN"))
    print("MILVUS_COLLECTION_NAME",os.getenv("MILVUS_COLLECTION_NAME"))
    print("EMBEDDING_MODEL",os.getenv("EMBEDDING_MODEL"))
    