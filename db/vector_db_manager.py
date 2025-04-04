import os
import shutil
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter, SimpleNodeParser, SemanticSplitterNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb
from chromadb.errors import InvalidCollectionException
from typing import List

class VectorDBManager:
    def __init__(self, db_dir: str, new_documents_dir: str, processed_documents_dir: str, db_name: str):
        self.db_dir = db_dir
        self.new_documents_dir = new_documents_dir
        self.processed_documents_dir = processed_documents_dir
        self.db_name = db_name
        self.client = chromadb.PersistentClient(path=db_dir)
        self.collection = self._get_or_create_collection()
        self.node_parser = SimpleNodeParser.from_defaults()

        # Ensure directories exist
        os.makedirs(self.new_documents_dir, exist_ok=True)
        os.makedirs(self.processed_documents_dir, exist_ok=True)

    def _get_or_create_collection(self):
        """Ensure collection exists and is not empty."""
        try:
            # Attempt to retrieve the collection
            collection = self.client.get_collection(self.db_name) #exists but is empty
            
            # Check if the collection is empty
            if collection.count() == 0:
                return collection 
            return collection  # Return the existing, non-empty collection
        
        except InvalidCollectionException:
            # If the collection doesn't exist, create it 
            return self.client.create_collection(self.db_name) 

        except Exception as e:
            # General exception catch for any other unforeseen errors
            print(f"An error occurred: {str(e)}")
            raise

    def get_new_files(self) -> List[str]:
        return [f for f in os.listdir(self.new_documents_dir)
                if os.path.isfile(os.path.join(self.new_documents_dir, f))]

    def process_new_documents(self) -> VectorStoreIndex:
        new_files = self.get_new_files()

        if not new_files:
            return None  # No new documents to process

        # Load and process new documents
        new_docs = SimpleDirectoryReader(
            input_files=[os.path.join(self.new_documents_dir, f) for f in new_files],
            filename_as_id=True
        ).load_data()        

        # Insert into index
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=70)
        
        # Semantic chunking
        # embed_model = OpenAIEmbedding()
        # splitter = SemanticSplitterNodeParser(
        #     buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model)
        
        nodes = splitter.get_nodes_from_documents(new_docs)
        vector_store = ChromaVectorStore(chroma_collection=self.collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context,show_progress=True)

        # Move processed files
        for filename in new_files:
            shutil.move(os.path.join(self.new_documents_dir, filename),
                        os.path.join(self.processed_documents_dir, filename))

        # Update the ChromaDB collection
        self.collection.add(
            ids=[doc.doc_id for doc in new_docs],
            documents=[doc.text for doc in new_docs],
            metadatas=[doc.metadata for doc in new_docs]
        )

        return index
