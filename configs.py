class Configs():
    PROCESSED_DOCUMENTS_DIR= './documents/processed' 
    NEW_DOCUMENTS_DIR= './documents/new' 
    DB_DIR= './chroma_db'
    EMBEDDING_MODEL= 'text-embedding-3-small'
    MODEL_NAME='gpt-4o-mini'
    TEMPERATURE=1
    MAX_TOKENS=1024 
    CACHE_DIR= './PIPELINE_DOCS'
    DB_NAME= 'knowledge_repository' 
    SIMILARITY_TOP_K=10
    SYSTEM_PROMPT="""
    You are a helpful assistant to answer students and teachers questions. 
    You are here to answer questions based on the context given.
    You are prohibited from using prior knowledge and you can only use the context given. 
    If you need more information, please ask the user.
    """
    
    CONTEXT_PROMPT_TEMPLATE = (
        "Context information:\n{context_str}\n"
        "---------------------\n"
        "Answer the question using this context."
    )