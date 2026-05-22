import os
import asyncio
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure the API key globally if present
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class GeminiClient:
    def __init__(self):
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    async def generate_response(self, prompt: str) -> str:
        """
        Calls Gemini API with retry logic and async execution.
        """
        model = genai.GenerativeModel(self.model_name)
        
        for attempt in range(1, 4):
            try:
                # Call Gemini async API
                response = await model.generate_content_async(prompt)
                return response.text
            except Exception as e:
                print(f"Gemini API Error (Attempt {attempt}): {e}", flush=True)
                if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < 3:
                    await asyncio.sleep([2, 4][attempt - 1])
                else:
                    raise e
        
        raise RuntimeError("Failed to generate response from Gemini after 3 attempts")
