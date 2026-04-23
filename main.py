from pipeline.daddy_pipeline import daddy_pipeline

if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(daddy_pipeline,package_path="daddy_pipeline.yaml")
