from kfp import dsl,compiler
import dotenv
dotenv.load_dotenv()
from coponents.llm_md import llm_md
from coponents.llm_seed_data import llm_seed_data
from coponents.llm_candidate_generator import llm_candidate_generator
from coponents.db_worker import db_worker
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
    c=db_worker(data=candidate_answers.output)
    c.set_caching_options(enable_caching=False)
    c.set_env_variable("DB_HOST", os.getenv("DB_HOST"))
    c.set_env_variable("DB_USER", os.getenv("DB_USER"))
    c.set_env_variable("DB_PASSWORD", os.getenv("DB_PASSWORD"))
    c.set_env_variable("DB_NAME", os.getenv("DB_NAME"))
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
    