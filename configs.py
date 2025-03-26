class Configs():
    DOCUMENTS_DIR= './documents' 
    DB_DIR= './chroma_db'
    EMBEDDING_MODEL= 'text-embedding-3-small'
    LLM_MODEL='gpt-4o-mini'
    TEMPERATURE=1
    MAX_TOKENS=512
    ARXIV_MAX_RESULTS = 10
    CACHE_DIR= './PIPELINE_DOCS'
    DB_NAME= 'knowledge_repository'
    BATCH_SIZE= 100 
    SIMILARITY_TOP_K=3
    SYSTEM_PROMPT="""You are a Q&A bot. You are here to answer questions based on the context given.
        You are prohibited from using prior knowledge and you can only use the context given. If you need 
        more information, please ask the user."""