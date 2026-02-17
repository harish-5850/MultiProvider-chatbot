import streamlit as st
import sys
import os

# Ensure the 'utils' folder can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.llm_service import LLMFactory

st.set_page_config(page_title="MVGR GenAI Lab", page_icon="🤖")
st.title("🤖 MVGR Chat Assistant")
st.caption("Module 3: Visualizing Real-time Streaming")

# 1. Initialize the provider
provider = LLMFactory.get_provider("gemini")

# 2. Setup Chat History (so it looks like a real app)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Handle User Input
if prompt := st.chat_input("Ask me anything about MVGR or GenAI..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 4. Handle Assistant Response with STREAMING
    with st.chat_message("assistant"):
        # st.write_stream is a magic function that handles Python generators!
        response_generator = provider.generate_stream(prompt)
        full_response = st.write_stream(response_generator)
    
    # Save the full response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
