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
        """
        Takes raw web text and turns it into a structured Markdown guide.
        """
        schema_json = FinalSynthesis.model_json_schema()
        
        # 1. Prepare the raw data block for the LLM
        sources_text = ""
        for idx, item in enumerate(scraped_data):
            clean_text = item['content'][:6000] 
            sources_text += f"\n--- SOURCE {idx + 1} ({item['url']}) ---\n{clean_text}\n"

        # 2. Craft the Synthesis Prompt
        system_prompt = f"""
        You are an Elite Gaming Meta Analyst and Professional Guide Writer.
        Your task is to analyze raw, messy web-scraped text and synthesize a perfect, highly-accurate gaming guide.
        
        USER QUERY: "{user_query}"
        TARGET LANGUAGE: {query_language}
        
        INSTRUCTIONS:
        1. READ THE RAW DATA: Ignore navigation menus, cookie warnings, and irrelevant text. Extract ONLY the actual stats, skill builds, rotations, or meta advice.
        2. VERIFY VERSION: Find the exact patch/version mentioned in the text (e.g., '17.0', 'Season 4'). If it's not explicitly stated, set 'verified_version' to "Unknown".
        3. WRITE THE GUIDE: Create a comprehensive, actionable guide formatted in clean Markdown. 
           - Use headings (##, ###), bullet points, and **bold text** for emphasis.
           - Group information logically (e.g., Core Skills, Stats Priority, Gear).
           - The guide MUST be written entirely in the TARGET LANGUAGE ({query_language}).
        4. NO HALLUCINATIONS: You are strictly bound by the provided SOURCE text. If the sources do not contain enough info to answer the query, explicitly state what is missing. DO NOT invent stats or game mechanics.
        
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
        """
        Handles follow-up questions using the previously scraped data and chat history.
        """
        # Re-build the context from the scraped data so the LLM remembers the facts
        sources_text = ""
        for idx, item in enumerate(scraped_data):
            clean_text = item['content'][:6000]
            sources_text += f"\n--- SOURCE {idx + 1} ({item['url']}) ---\n{clean_text}\n"

        system_prompt = f"""
        You are an Elite Gaming Meta Analyst assisting a user with follow-up questions.
        You previously generated a meta guide based on specific web-scraped data.
        
        INSTRUCTIONS:
        1. Answer the user's new question using ONLY the provided SOURCE DATA.
        2. If the answer cannot be found in the source data, politely explain that the current data doesn't contain that specific information. DO NOT hallucinate stats.
        3. Maintain a helpful, expert, and professional tone.
        4. Format your response in clean Markdown.
        
        SOURCE DATA TO BASE YOUR ANSWERS ON:
        {sources_text}
        """

        # 1. Start with the system prompt
        messages = [{"role": "system", "content": system_prompt}]
        
        # 2. Append the previous conversation history
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
        # 3. Add the user's brand new question
        messages.append({"role": "user", "content": new_message})

        # Call the model (Notice we do NOT force JSON format here, just a standard text reply)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3
        )
        
        # Return the plain text (Markdown) response
        return completion.choices[0].message.content