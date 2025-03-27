import chromadb
import chromadb.errors
from chromadb.errors import InvalidCollectionException
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from typing import Optional, List, Dict

from dotenv import load_dotenv
import os
import shutil
from pathlib import Path

from configs import Configs

load_dotenv()
# os.environ['OPENAI_API_KEY'] = Configs.OPENAI_API_KEY


class VectorDBManager:
    """Manages ChromaDB collection using file movement as processing tracker"""
    def __init__(self, db_dir: str, new_documents_dir: str, processed_documents_dir: str, db_name: str):
        self.db_dir = db_dir
        self.new_documents_dir = new_documents_dir
        self.processed_documents_dir = processed_documents_dir
        self.db_name = db_name
        self.client = chromadb.PersistentClient(path=db_dir)
        self.collection = self._get_or_create_collection()
        self.node_parser = SimpleNodeParser.from_defaults()

        # Ensure directories exist
        Path(self.new_documents_dir).mkdir(parents=True, exist_ok=True)
        Path(self.processed_documents_dir).mkdir(parents=True, exist_ok=True)

    def _get_or_create_collection(self):
        """Simplified collection initialization""" 
        try:
            collection = self.client.get_collection(self.db_name)
            if collection.count() == 0:
                raise ValueError("Collection exists but is empty")
            return collection
        except (chromadb.errors.InvalidCollectionException, ValueError):
            return self.client.create_collection(self.db_name)

    def get_new_files(self) -> List[str]:
        """Get all files in the new documents directory"""
        return [
            f for f in os.listdir(self.new_documents_dir)
            if os.path.isfile(os.path.join(self.new_documents_dir, f))
        ]

    def create_index(self, documents: list) -> VectorStoreIndex:
        """Create index with automatic ID management"""
        vector_store = ChromaVectorStore(chroma_collection=self.collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )

    def update_index(self, index: VectorStoreIndex) -> None:
        """Process all new documents and move them to processed directory"""
        new_files = self.get_new_files()
        
        if new_files:
            # Load and process all new documents
            new_docs = SimpleDirectoryReader(
                input_files=[os.path.join(self.new_documents_dir, f) for f in new_files],
                filename_as_id=True
            ).load_data()

            # Insert into index
            nodes = self.node_parser.get_nodes_from_documents(new_docs)
            index.insert_nodes(nodes)
            
            # Update ChromaDB collection
            self.collection.add(
                ids=[doc.doc_id for doc in new_docs],
                documents=[doc.text for doc in new_docs],
                metadatas=[doc.metadata for doc in new_docs]
            )
            
            # Move processed files
            for filename in new_files:
                src = os.path.join(self.new_documents_dir, filename)
                dest = os.path.join(self.processed_documents_dir, filename)
                shutil.move(src, dest)

class Retriever:
    """Handles document retrieval and context formatting"""
    def __init__(self, index: VectorStoreIndex):
        self.index = index
        self.prompt_template = Configs.CONTEXT_PROMPT_TEMPLATE

    def retrieve(self, query: str, top_k: int = Configs.SIMILARITY_TOP_K) -> list:
        """Retrieve relevant nodes for a query"""
        return self.index.as_retriever(similarity_top_k=top_k).retrieve(query)

    def format_context(self, nodes: list) -> str:
        """Format retrieved nodes into context string"""
        context_str = "\n\n".join(
            [n.node.get_content(metadata_mode='all') for n in nodes]
        )
        return self.prompt_template.format(context_str=context_str)


class ChatEngine:
    """Manages chat interactions and LLM communication"""
    def __init__(self, llm: OpenAI, retriever: Retriever):
        self.llm = llm
        self.retriever = retriever
        self.system_prompt = Configs.SYSTEM_PROMPT

    def _format_messages(self, query: str, history: list, use_context: bool) -> List[ChatMessage]:
        """Prepare message list for LLM"""
        messages = [ChatMessage(role="system", content=self.system_prompt)]
        
        for msg in history:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append(ChatMessage(role=role, content=msg["content"]))
        
        if use_context:
            nodes = self.retriever.retrieve(query)
            context = self.retriever.format_context(nodes)
            query = f"{context}\n\nQuestion: {query}"
        
        messages.append(ChatMessage(role="user", content=query))
        return messages

    def chat(self, query: str, history: list = None, use_context: bool = True) -> str:
        """Process chat request"""
        history = history or []
        messages = self._format_messages(query, history, use_context)
        response = self.llm.chat(messages)
        return str(response).strip()


class Generator:
    """Main interface with simplified initialization"""
    def __init__(self):
        self.config = Configs()
        
        self.vector_db = VectorDBManager(
            db_dir=self.config.DB_DIR,
            new_documents_dir=self.config.NEW_DOCUMENTS_DIR,
            processed_documents_dir=self.config.PROCESSED_DOCUMENTS_DIR,
            db_name=self.config.DB_NAME
        )
        self.index = self._initialize_index()
        self.retriever = Retriever(self.index)
        self.llm = OpenAI(model=self.config.MODEL_NAME, temperature=0)
        self.chat_engine = ChatEngine(self.llm, self.retriever)

    def _initialize_index(self) -> VectorStoreIndex:
        """Initialize or load index with automatic file handling"""
        try:
            # Load existing index
            index = VectorStoreIndex.from_vector_store(
                ChromaVectorStore(self.vector_db.collection),
                storage_context=StorageContext.from_defaults(
                    vector_store=ChromaVectorStore(self.vector_db.collection)
                )
            )
            # Process any new files that arrived before initialization
            self.vector_db.update_index(index)
            return index
        except Exception:
            # Create new index from all files in new_documents_dir
            documents = SimpleDirectoryReader(
                self.vector_db.new_documents_dir,
                filename_as_id=True
            ).load_data()
            
            index = self.vector_db.create_index(documents)
            
            # Move initial batch to processed directory
            for filename in os.listdir(self.vector_db.new_documents_dir):
                src = os.path.join(self.vector_db.new_documents_dir, filename)
                dest = os.path.join(self.vector_db.processed_documents_dir, filename)
                shutil.move(src, dest)
            
            return index 

    def chat(self, query: str, chat_history: list = None, use_context: bool = True) -> str:
        """Public interface for chat functionality"""
        return self.chat_engine.chat(query, chat_history, use_context)