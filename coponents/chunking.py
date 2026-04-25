from kfp import dsl

@dsl.component(
    base_image="python:3.12",
    packages_to_install=["sentence-transformers>=5.4.1","pymilvus>=2.6.12","python-dotenv>=1.2.2"],
)
def chunking(md:str)->str:
    import os
    import json 
    import uuid
    import dotenv
    from pymilvus import MilvusClient,DataType
    from sentence_transformers import SentenceTransformer
    dotenv.load_dotenv()
    client_milvus = MilvusClient(
        uri=os.getenv("MILVUS_URI"),
        token=os.getenv("MILVUS_TOKEN")
    )
    collection_name=os.getenv("MILVUS_COLLECTION_NAME")
    embedding_model=os.getenv("EMBEDDING_MODEL")
    embedd=SentenceTransformer(embedding_model)
    dim=embedd.get_embedding_dimension()
    if not client_milvus.has_collection(collection_name):
        schema=client_milvus.create_schema(auto_id=False)
        schema.add_field(field_name="chunk_id",datatype=DataType.VARCHAR,is_primary=True,max_length=100)
        schema.add_field(field_name="text",datatype=DataType.VARCHAR,max_length=2000)
        schema.add_field(field_name="embedding",datatype=DataType.FLOAT_VECTOR,dim=dim)
        index_params=client_milvus.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        client_milvus.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        client_milvus.load_collection(collection_name)
    text=md.replace("\r\n","\n").strip()
    paras=[p.strip() for p in text.split("\n\n") if p.strip()]
    chunks=[]
    current=""
    maxi_chars=1000
    for i in paras:
        if current=="":
            current=i
        elif len(current)+len(i)<=maxi_chars:
            current+=" "+i
        else:
            chunks.append(current)
            current=i
    if current:
        chunks.append(current)
    
    # if current!="":
    #     chunks.append(current)
    vectors=embedd.encode(chunks,normalize_embeddings=True,show_progress_bar=True).tolist()
    ans=[]
    for i,j in zip(chunks,vectors):
        ans.append({
            "chunk_id":str(uuid.uuid4()),
            "text":i,
            "embedding":j
        })
    client_milvus.insert(collection_name=collection_name,data=ans)
    client_milvus.flush(collection_name=collection_name)
    client_milvus.load_collection(collection_name)
    return json.dumps(ans,ensure_ascii=False)


######semantic logic

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
    