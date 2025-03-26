import chromadb

import chromadb.errors
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import SimpleDirectoryReader 
from llama_index.core import PromptTemplate    
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI

from typing import Optional, Tuple
from typing import List, Dict

from dotenv import load_dotenv
import os

from configs import Configs

load_dotenv()
# os.environ['OPENAI_API_KEY'] = Configs.OPENAI_API_KEY
print(f"OPENAI_API_KEY: " + os.environ['OPENAI_API_KEY'])

class Generator:
    def __init__(self, path: Optional[str] = None):
        self.path = path or Configs.DOCUMENTS_DIR
        self.index = self.create_or_get_vector_db()
        self.query_engine: Optional[BaseChatEngine] = None
        self.llm = OpenAI(model="gpt-4o-mini", temperature=0)

    def create_or_get_vector_db(self) -> VectorStoreIndex:
        """Create or load existing vector database index."""
        db = chromadb.PersistentClient(path=Configs.DB_DIR)
        
        try:
            chroma_collection = db.get_collection(name=Configs.DB_NAME)
            if chroma_collection.count() == 0:
                raise ValueError("Collection exists but is empty")
            
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
            
        except (chromadb.errors.InvalidCollectionException, ValueError):
            documents = SimpleDirectoryReader(self.path, filename_as_id=True).load_data()
            chroma_collection = db.get_or_create_collection(name=Configs.DB_NAME)
            
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            return VectorStoreIndex.from_documents(
                documents, 
                storage_context=storage_context,
                show_progress=True
            )

    def retriever(self, user_query: str, similarity_top_k: int = Configs.SIMILARITY_TOP_K):
        """Retrieve relevant nodes for a query."""
        return self.index.as_retriever(similarity_top_k=similarity_top_k).retrieve(user_query)
     
    def build_context_prompt(self, retrieved_nodes) -> Tuple[str, list]:
        """Build context prompt from retrieved nodes."""
        context_prompt = PromptTemplate(
            "Context information to answer the query is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
        )
        context_str = "\n\n".join([r.node.get_content(metadata_mode='all') for r in retrieved_nodes])
        return context_prompt.format(context_str=context_str), retrieved_nodes

    def context_retriever(self, user_query: str, similarity_top_k: int = Configs.SIMILARITY_TOP_K):
        """Retrieve context and formatted prompt."""
        return self.build_context_prompt(self.retriever(user_query, similarity_top_k))

    def chat(self, user_query: str, chat_history: List[Dict] = None, use_context: bool = True) -> str:
        """Handle chat with context and history awareness."""
        messages = []
        
        # System message setup
        messages.append(ChatMessage(role="system", content=Configs.SYSTEM_PROMPT))
        
        # Process chat history
        if chat_history:
            for msg in chat_history:
                role = "user" if msg["role"] == "user" else "assistant"
                messages.append(ChatMessage(role=role, content=msg["content"]))
        
        # Add context if enabled
        if use_context:
            context_prompt, _ = self.context_retriever(user_query)
            messages.append(ChatMessage(role="user", content=context_prompt))
         
        # Initialize query engine if needed
        if not self.query_engine:
            self.query_engine = self.index.as_chat_engine(
                chat_mode="condense_question",
                llm=self.llm,
                verbose=True,
                streaming=True,
                similarity_top_k=Configs.SIMILARITY_TOP_K
            )
        
        # Generate response
        response = self.query_engine.chat(user_query)
        return str(response)