import google.generativeai as genai #type: ignore
import os

from dotenv import load_dotenv #type: ignore
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-flash-latest")

def process_sentence(words: str) -> str:
    try:
        prompt = f"""You are a grammar correction assistant for American Sign Language (ASL).
Convert this ASL gloss to natural English. Add missing function words and fix word order.
Reply ONLY with the corrected sentence.

ASL gloss: {words}"""

        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return words