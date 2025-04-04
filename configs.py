class Configs():
    LOG_LEVEL="INFO"
    WRITE_LOGS=True

    PROCESSED_DOCUMENTS_DIR= './documents/processed' 
    NEW_DOCUMENTS_DIR= './documents/new' 
    VECTOR_DB_DIR= './chroma_db'
    SQLITE_DB_DIR = "./db/sqlite_data.db"


    EMBEDDING_MODEL= 'text-embedding-3-small'
    MODEL_NAME='gpt-4o-mini'
    TEMPERATURE=1
    MAX_TOKENS=1024 
    CACHE_DIR= './PIPELINE_DOCS'
    DB_NAME= 'knowledge_repository' 
    SIMILARITY_TOP_K=10
    SYSTEM_PROMPT="""
    You are a helpful assistant to answer students and teachers questions. 
    You are here to answer questions based on the documents given.
    You are prohibited from using prior knowledge and you can only use the documents given. 
    If you need more information, please ask the user.
    """
    
    CONTEXT_PROMPT_TEMPLATE = (
        "documents:\n{context_str}\n"
        "---------------------\n"         
    )

    REWRITE_QUERY_PROMPT = """
    Given a conversation (between Human and Assistant) and a follow up message from Human, 
    rewrite the message to be a standalone question that captures all relevant information 
    from the conversation. 
    Sometimes, human may say a greeting, initiate a conversation, or conclude a conversation, none of which require a standalone question
    """    

    REWRITE_QUERY_TEMPLATE = """ 
    <Chat History>
    {chat_history}
    <Follow Up Message>
    {question}
    <Standalone question>
    assistant:
    """