import os
import json
from datetime import datetime
from groq import Groq
from app.api.schemas import RouterResponse

class RouterAgent:
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model

    def generate_search_plan(self, user_query: str) -> RouterResponse:
        schema_json = RouterResponse.model_json_schema()
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_year = datetime.now().year 
        
        system_prompt = f"""
You are a Research Architect specializing in Gaming Intelligence. 
Today's Date: {current_date}. Current Year: {current_year}.

CRITICAL RULE: TARGET FOCUS
If the user asks about a specific character, class, weapon, or strategy, that EXACT SUBJECT MUST be included in EVERY SINGLE search query.

CRITICAL RULE: CIS SLANG TRANSLATION
Sometimes gamers use heavy using slang and abbreviations. You MUST translate slang to the OFFICIAL GLOBAL GAME NAME before generating search queries. 
Examples of common CIS slang:
- "мобла", "млбб", "мл" -> "Mobile Legends: Bang Bang"
- "алоды", "алодах", "аллоды" -> "Аллоды Онлайн" (Allods Online)
- "дока", "дота", "дотка" -> "Dota 2"
- "кс", "ксго", "контра" -> "Counter-Strike 2"
- "вов", "wow", "вовка" -> "World of Warcraft"
- "лига", "лол", "lol" -> "League of Legends"
- "пабг", "пубг" -> "PUBG: Battlegrounds"
- "тарков", "ефт" -> "Escape from Tarkov"

If the user writes "билд на эстеса в мобле", your queries MUST use "Mobile Legends Estes...", NOT "мобла Estes...".

SEARCH QUERY LOGIC (80/20 ENGLISH STRATEGY):
You MUST generate EXACTLY 5 queries to get the most hardcore, technical data:
1. 80% ENGLISH QUERIES (Generate 4 queries in English):
   - Query 1: Find hardcore math and stats (e.g., "[Subject] build exact stats math multipliers {current_year}").
   - Query 2: Find the latest English patch notes (e.g., "[Subject] patch notes nerfs buffs {current_year}").
   - Query 3: Find pro/global meta tier list (e.g., "[Subject] global pro build {current_year}").
   - Query 4: Find exact item names and skill mechanics (e.g., "[Subject] wiki skills items {current_year}").
2. 20% LOCAL QUERY (Generate 1 query in the USER'S LANGUAGE):
   - To capture local community tips (e.g., "Гайд [Subject] {current_year} советы").
3. ANTI-SEO: Use negative keywords in DDG: e.g., "-2022 -2023 -2024".

OUTPUT FORMAT:
- Output ONLY valid JSON matching this schema:
{json.dumps(schema_json, indent=2)}
"""

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        return RouterResponse.model_validate_json(completion.choices[0].message.content)