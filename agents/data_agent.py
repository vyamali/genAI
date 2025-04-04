from typing import List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from db.sqlite import SQLiteDatabase  # Assuming this is your SQLiteDatabase class

class SqliteAgent:
    def __init__(self, db_file: str, model_name: str = "gpt-4o-mini", temperature: float = 0):
        """
        Initialize the SQLAgent with a connection to the SQLite database and a language model.

        Args:
            db_file (str): Path to the SQLite database file.
            model_name (str): OpenAI model name to use (default is "gpt-4o-mini").
            temperature (float): Temperature for the language model (default is 0).
        """
        # 1. Initialize Database Connection using the provided SQLiteDatabase class
        self.db = SQLiteDatabase(db_file)

        # 2. Initialize Language Model
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

        # 3. Create SQL Database Tools
        @tool
        def list_tables_tool():
            """List all tables in the database."""
            return self.db.list_tables()

        @tool
        def describe_table_schema(table_name: str):
            """Get the schema for a specific table."""
            return self.db.describe_table(table_name)

        @tool
        def execute_sql_query(query: str):
            """Execute a SQL query and return results."""
            try:
                return self.db.execute_query(query)
            except Exception as e:
                return f"Error executing query: {str(e)}"

        # 4. Combine Tools
        self.tools = [
            list_tables_tool,
            describe_table_schema,
            execute_sql_query
        ]

        # 5. System Prompt for SQL Agent
        self.system_message = """ 
        You are an expert SQL database assistant. 
        You will take the users questions and turn them into SQL queries using the tools available.
        Follow these guidelines:
        1. Always start by understanding the database structure
        2. Break down complex questions into step-by-step reasoning
        3. Use tools to explore the database schema
        4. Generate precise, efficient SQL queries
        5. Handle errors gracefully
        6. Provide clear explanations of your reasoning

        When solving a problem:
        - First, list available tables using list_tables 
        - Then, examine relevant table schemas using describe_table 
        - Generate an appropriate SQL query
        - Execute the query and interpret results using execute_query 
        - Provide a human-readable answer
        """

        # 6. Create React Agent
        self.agent_executor = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=self.system_message
        )

    def sql_agent_query(self, question: str) -> dict:
        """
        Execute a SQL query using the ReAct agent

        Args:
            question (str): Natural language question about the database

        Returns:
            dict: Agent's response with reasoning and answer
        """
        response = self.agent_executor.invoke({
            "messages": [HumanMessage(content=question)]
        })
        
        return response 