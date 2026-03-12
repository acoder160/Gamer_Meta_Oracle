import os
import json
from groq import Groq
from typing import List, Dict
from app.api.schemas import FinalSynthesis

class SynthesizerAgent:
    """
    Phase 3 Agent: The Brain. 
    Reads raw scraped data, synthesizes meta guides, and handles follow-up Q&A.
    """
    
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model

    def generate_guide(self, user_query: str, query_language: str, scraped_data: List[Dict[str, str]]) -> FinalSynthesis:
        schema_json = FinalSynthesis.model_json_schema()
        
        # 1. Prepare the raw data block for the LLM with a STRICT TPM LIMIT
        sources_text = ""
        max_chars = 28000 # Безопасный лимит (около 7000-8000 токенов), чтобы оставить место для ответа
        
        for idx, item in enumerate(scraped_data):
            clean_text = item['content'][:4000] # Берем до 4000 символов с каждого сайта
            
            # Если добавление этого сайта превысит лимит - останавливаемся
            if len(sources_text) + len(clean_text) > max_chars:
                print(f"[*] Context limit reached. Used {idx} out of {len(scraped_data)} sources.")
                break
                
            sources_text += f"\n--- SOURCE {idx + 1} ({item['url']}) ---\n{clean_text}\n"

        # 2. Craft the Synthesis Prompt
        system_prompt = f"""
        You are an Elite Gaming Meta Analyst and Professional Guide Writer.
        Your task is to analyze raw, messy web-scraped text and synthesize a perfect, highly-accurate gaming guide.
        
        USER QUERY: "{user_query}"
        TARGET LANGUAGE: {query_language}
        
        INSTRUCTIONS:
        1. READ THE RAW DATA: Ignore menus and irrelevant text. Extract ONLY stats, skill builds, rotations, or meta.
        2. VERIFY VERSION: Find the exact patch/version mentioned. If not stated, set 'verified_version' to "Unknown".
        3. WRITE THE GUIDE: Create a comprehensive guide formatted in clean Markdown. 
           - Use headings (##, ###), bullet points, and **bold text**.
           - The guide MUST be written entirely in the TARGET LANGUAGE ({query_language}).
        4. NO HALLUCINATIONS: You are strictly bound by the provided SOURCE text.
        
        OUTPUT FORMAT:
        Output strictly valid JSON matching this schema:
        {json.dumps(schema_json, indent=2)}
        """

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the raw scraped data to analyze:\n{sources_text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.3 
        )
        
        return FinalSynthesis.model_validate_json(completion.choices[0].message.content)

    def continue_chat(self, new_message: str, chat_history: List[Dict[str, str]], scraped_data: List[Dict[str, str]]) -> str:
        # Умный лимит и для чата тоже
        sources_text = ""
        max_chars = 25000 
        
        for idx, item in enumerate(scraped_data):
            clean_text = item['content'][:3000]
            if len(sources_text) + len(clean_text) > max_chars:
                break
            sources_text += f"\n--- SOURCE {idx + 1} ({item['url']}) ---\n{clean_text}\n"

        system_prompt = f"""
        You are an Elite Gaming Meta Analyst assisting a user with follow-up questions.
        You previously generated a meta guide based on specific web-scraped data.
        
        INSTRUCTIONS:
        1. Answer the user's new question using ONLY the provided SOURCE DATA.
        2. If the answer cannot be found, explain that the current data doesn't contain it. DO NOT hallucinate.
        3. Format your response in clean Markdown.
        
        SOURCE DATA TO BASE YOUR ANSWERS ON:
        {sources_text}
        """

        messages = [{"role": "system", "content": system_prompt}]
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": new_message})

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3
        )
        return completion.choices[0].message.content