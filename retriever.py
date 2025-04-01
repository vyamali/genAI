from llama_index.core import VectorStoreIndex
from configs import Configs

class Retriever:
    def __init__(self, index: VectorStoreIndex):
        self.index = index
        self.prompt_template = Configs.CONTEXT_PROMPT_TEMPLATE

    def retrieve(self, query: str, top_k: int = Configs.SIMILARITY_TOP_K) -> list:
        return self.index.as_retriever(similarity_top_k=top_k).retrieve(query)

    def format_context(self, nodes: list) -> str:
        context_str = "\n\n".join([n.node.get_content(metadata_mode='all') for n in nodes])
        return self.prompt_template.format(context_str=context_str)
