from db.vector_db_manager import VectorDBManager
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from retriever import Retriever
from chat_engine import ChatEngine
from agents.data_agent import SqliteAgent
from llama_index.llms.openai import OpenAI
from configs import Configs

from dotenv import load_dotenv

class Generator:
    def __init__(self):
        self.config = Configs()
        load_dotenv()

        # Initialize VectorDBManager
        self.vector_db = VectorDBManager(
            db_dir=self.config.VECTOR_DB_DIR,
            new_documents_dir=self.config.NEW_DOCUMENTS_DIR,
            processed_documents_dir=self.config.PROCESSED_DOCUMENTS_DIR,
            db_name=self.config.DB_NAME
        )

        # Initialize or load the index
        self.index = self._initialize_index()

        # Initialize Retriever and ChatEngine
        self.retriever = Retriever(self.index)
        self.llm = OpenAI(model=self.config.MODEL_NAME, temperature=0.5)
        self.chat_engine = ChatEngine(self.llm, self.retriever)
        self.sql_agent = SqliteAgent(db_file=Configs.SQLITE_DB_DIR)

    def _initialize_index(self) -> VectorStoreIndex:
        """Initialize or load index with automatic file handling"""
        index = self.vector_db.process_new_documents()

        if index is None:
            # If no new documents, load the existing index
            index = VectorStoreIndex.from_vector_store(
                ChromaVectorStore(self.vector_db.collection),
                storage_context=StorageContext.from_defaults(
                    vector_store=ChromaVectorStore(self.vector_db.collection)
                )
            )
        return index

    def chat(self, query: str, chat_history: list = None, use_context: bool = True) -> str:
        """Public interface for chat functionality"""
        return self.chat_engine.chat(query, chat_history, use_context)

    def sql_query(self, query: str, chat_history: list = None, use_context: bool = True) -> str:
        """Public interface for chat functionality"""
        response = self.sql_agent.sql_agent_query(query)
        return response['messages'][-1].content