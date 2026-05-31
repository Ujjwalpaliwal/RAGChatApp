from google import genai
from typing import List, Dict

class GeminiRAGClient:
    def __init__(self, api_key: str):
        """Initialize modern google-genai client."""
        if not api_key:
            raise ValueError("Google Gemini API Key is required.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = 'gemini-2.5-flash'
        
    def generate_response(self, prompt: str) -> str:
        """Call Gemini model to generate text response."""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Error generating response from Gemini: {str(e)}"
