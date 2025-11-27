import os
import google.generativeai as genai
from dotenv import load_dotenv
from retriever import DocumentRetriever
from utils import format_chat_history
from typing import List, Dict, Any

load_dotenv()

class RAGChatbot:
    def __init__(self, gemini_api_key: str = None):
        # Initialize Gemini
        self.api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY environment variable or pass it directly.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Initialize retriever
        self.retriever = DocumentRetriever()
        
        # Chat history
        self.history: List[Dict[str, str]] = []
    
    def _create_prompt(self, query: str, context: str, history: str) -> str:
        """Create the RAG prompt for Gemini."""
        prompt = f"""You are a helpful assistant that answers questions based on the provided context and conversation history.

CONVERSATION HISTORY:
{history}

RETRIEVED CONTEXT:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer the question primarily based on the retrieved context
2. If the context doesn't contain relevant information, say so and provide a general answer if possible
3. Maintain conversation flow considering the history
4. Be concise but thorough
5. Cite which documents you're referencing when appropriate

ANSWER:"""
        return prompt
    
    def chat(self, query: str, use_history: bool = True) -> str:
        """Main chat method with RAG."""
        # Retrieve relevant documents
        context = self.retriever.get_context(query)
        
        # Format history if needed
        history_text = format_chat_history(self.history) if use_history else ""
        
        # Create prompt
        prompt = self._create_prompt(query, context, history_text)
        
        try:
            # Generate response
            response = self.model.generate_content(prompt)
            answer = response.text
            
            # Update history
            self.history.append({"role": "user", "content": query})
            self.history.append({"role": "assistant", "content": answer})
            
            return answer
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def clear_history(self):
        """Clear chat history."""
        self.history.clear()

# Singleton instance
_chatbot_instance = None

def get_chatbot(api_key: str = None) -> RAGChatbot:
    """Get or create chatbot instance."""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = RAGChatbot(api_key)
    return _chatbot_instance