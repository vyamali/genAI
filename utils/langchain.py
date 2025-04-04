from langchain_core.messages import HumanMessage,AIMessage, ToolMessage 

class Langchain:

    def __init__(self):
        pass

    def read_detailed_response(self, response):
        """Comprehensive response parsing"""
        print("\n=== Detailed Response Breakdown ===")
        
        # Iterate through all messages
        for i, message in enumerate(response['messages']):
            print(f"\nMessage {i + 1}:")
            
            # Message Type Handling
            if isinstance(message, HumanMessage):
                print("User Query:", message.content)
            
            elif isinstance(message, AIMessage):
                # Check for tool calls
                if message.tool_calls:
                    print("Reasoning Step:")
                    for tool_call in message.tool_calls:
                        print(f"- Tool: {tool_call['name']}")
                        print(f"  Arguments: {tool_call.get('args', 'N/A')}")
                
                print("AI Thought:", message.content)
            
            elif isinstance(message, ToolMessage):
                print("Tool Result:", message.content)