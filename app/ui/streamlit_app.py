import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.search.vector_store import FAISSVectorStore
from src.search.embedder import DocumentEmbedder
from src.llm.gemini_client import GeminiRAGClient

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Premium UI CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(45deg, #1f77b4, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
    }
    .chat-container {
        background-color: #f7f9fc;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 12px 18px;
        border-radius: 18px 18px 2px 18px;
        margin: 8px 0;
        border-left: 4px solid #1e88e5;
    }
    .assistant-message {
        background-color: #f5f5f5;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 2px;
        margin: 8px 0;
        border-left: 4px solid #757575;
    }
    .source-tag {
        font-size: 0.8rem;
        color: #0d47a1;
        background-color: #e3f2fd;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

def initialize_components():
    """Load FAISS store, embedder, and return components."""
    # Initialize embedder and FAISS store
    embedder = DocumentEmbedder()
    vector_store = FAISSVectorStore()
    
    if not vector_store.load():
        st.warning("⚠️ FAISS vector index not built yet or not found. Please build the index first using `python src/build_index.py`.")
        return None, None
        
    return embedder, vector_store

def create_rag_prompt(query: str, context: str, history: list) -> str:
    """Format conversation context and history for the RAG prompt."""
    history_text = ""
    for msg in history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""You are a helpful assistant that answers questions based on the provided context and conversation history.

CONVERSATION HISTORY:
{history_text}

RETRIEVED CONTEXT:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer the question primarily based on the retrieved context.
2. If the context doesn't contain relevant information, state so, and provide a helpful general answer.
3. Maintain conversation flow considering the history.
4. Cite the source files when appropriate.

ANSWER:"""
    return prompt

def main():
    st.markdown('<div class="main-header">🧠 Modular RAG Chatbot</div>', unsafe_allow_html=True)
    
    # Sidebar config
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.getenv('GEMINI_API_KEY', ''),
            help="Enter your Google Gemini API key"
        )
        
        if api_key:
            st.session_state.api_key = api_key
        
        st.markdown("---")
        st.subheader("💡 About")
        st.write("This RAG chatbot uses:")
        st.write("• **FAISS** for document similarity search")
        st.write("• **Sentence-Transformers** for dense embeddings")
        st.write("• **Google Gemini (new SDK)** for response generation")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
            
    # Load model and vector DB
    embedder, vector_store = initialize_components()
    if not embedder or not vector_store:
        return
        
    # Get Gemini client
    api_key_to_use = st.session_state.get('api_key') or os.getenv('GEMINI_API_KEY')
    if not api_key_to_use:
        st.info("🔑 Please enter your Gemini API Key in the sidebar to begin.")
        return
        
    try:
        gemini_client = GeminiRAGClient(api_key=api_key_to_use)
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        return

    # Initialize message list
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    # Draw chat container
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Accept user prompt
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Retrieving document context and generating response..."):
            # 1. Retrieve
            query_emb = embedder.encode([prompt])
            results = vector_store.query(query_emb, k=3)
            
            # Format context
            context_parts = []
            sources = set()
            for idx, doc in enumerate(results, 1):
                source_name = os.path.basename(doc['source'])
                sources.add(source_name)
                context_parts.append(f"[Doc {idx} - {source_name}]:\n{doc['content']}")
                
            context = "\n\n".join(context_parts) if context_parts else "No context found."
            
            # 2. Generate
            rag_prompt = create_rag_prompt(prompt, context, st.session_state.messages[:-1])
            answer = gemini_client.generate_response(rag_prompt)
            
            # Add sources citation in UI if documents were matched
            if sources:
                source_list = " ".join([f'<span class="source-tag">{s}</span>' for s in sources])
                answer += f"\n\n*Sources:* {source_list}"
                
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        st.rerun()

if __name__ == "__main__":
    main()
