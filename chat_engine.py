from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from configs import Configs
from retriever import Retriever
import logging

class ChatEngine:
    def __init__(self, llm: OpenAI, retriever: Retriever):
        self.llm = llm
        self.retriever = retriever
        self.system_prompt = Configs.SYSTEM_PROMPT
        logging.basicConfig(level=Configs.LOG_LEVEL)

    def _rewrite_query_with_history(self, query: str, history: list, context_window: int = 5) -> str:
        if not history:
            return query
        recent_history = history[-context_window:]
        chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
        formatted_template = Configs.REWRITE_QUERY_TEMPLATE.format(chat_history=chat_history, question=query)
        rewrite_prompt = [
            ChatMessage(role="system", content=Configs.REWRITE_QUERY_PROMPT),
            ChatMessage(role="user", content=formatted_template)
        ]
        response = self.llm.chat(rewrite_prompt)
        return str(response).strip()

    def _format_messages(self, query: str, history: list, use_context: bool) -> list:
        messages = [ChatMessage(role="system", content=self.system_prompt)]  
        for msg in history:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append(ChatMessage(role=role, content=msg["content"]))
        if len(history) > 1:
            query = self._rewrite_query_with_history(query, history)
            if Configs.WRITE_LOGS:
                logging.info(f"rewrite_query_with_history: {query}")             

        if use_context:
            nodes = self.retriever.retrieve(query)
            context = self.retriever.format_context(nodes)
            query = f"{context}\n\nQuestion: {query}"
        messages.append(ChatMessage(role="user", content=query))
        return messages

    def chat(self, query: str, history: list = None, use_context: bool = True) -> str:
        history = history or []
        messages = self._format_messages(query, history, use_context)

        if Configs.WRITE_LOGS:
            logging.info(f"================================================================================\n") 
            logging.info(f"User query: {query}") 
            # logging.info(f"Message to LLM : {messages[-1]}")            

        response = self.llm.chat(messages)
        return str(response).strip()
