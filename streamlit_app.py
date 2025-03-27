import streamlit as st
from generator import Generator 

st.set_page_config(page_title="GenAI Agentic Application")

response_generator = Generator()
st.title("CampusConnect: AI-Powered Student Discovery 🧑💻")

# Initialize chat history with a welcome message
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome! How can I assist you today?"}
    ]

# Display chat messages with persistent history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input and processing
if prompt := st.chat_input("Your question"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            # Get conversation history (excluding current prompt)
            chat_history = st.session_state.messages[:-1]
            
            # Generate response with context and history
            response = response_generator.chat(
                query=prompt,
                chat_history=chat_history
            )
            
            # Display and store response
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})