import streamlit as st
import os
from dotenv import load_dotenv
import sys
import google.generativeai as genai

# Add src to path
sys.path.append('../src')

from chatbot import get_chatbot

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .user-message {
        background-color: #d1ecf1;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .assistant-message {
        background-color: #e2e3e5;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_chatbot():
    """Initialize the chatbot with API key."""
    api_key = st.session_state.get('api_key') or os.getenv('GEMINI_API_KEY')
    if not api_key:
        st.error("Please provide a Gemini API key")
        return None
    
    try:
        return get_chatbot(api_key)
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        return None

def main():
    st.markdown('<div class="main-header">🧠 RAG Chatbot with Gemini</div>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.getenv('GEMINI_API_KEY', ''),
            help="Enter your Google Gemini API key"
        )
        
        if api_key:
            st.session_state.api_key = api_key
        
        st.markdown("---")
        st.subheader("About")
        st.write("This RAG chatbot uses:")
        st.write("• FAISS for document retrieval")
        st.write("• Gemini for response generation")
        st.write("• Context from provided documents")
        
        if st.button("Clear Chat History"):
            if 'chatbot' in st.session_state:
                st.session_state.chatbot.clear_history()
            st.session_state.messages = []
            st.rerun()
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'chatbot' not in st.session_state:
        chatbot = initialize_chatbot()
        if chatbot:
            st.session_state.chatbot = chatbot
    
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about Chartered Accountants or Startups..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        if 'chatbot' in st.session_state:
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.chat(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("Chatbot not initialized. Please check your API key.")
        
        st.rerun()

if __name__ == "__main__":
    main()