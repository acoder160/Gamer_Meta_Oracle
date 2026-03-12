import os
import json
from datetime import datetime
from groq import Groq
from app.api.schemas import RouterResponse

class RouterAgent:
    """
    Phase 1 Agent: Responsible for analyzing game origin and 
    generating a multi-layered search strategy without hallucinations.
    """
    
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        # We assume GROQ_API_KEY is already in the environment
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model

    def generate_search_plan(self, user_query: str) -> RouterResponse:
        """
        Generates a search plan based on the user's query using zero-knowledge 
        logic to avoid version hallucinations.
        """
        # Get the technical schema to guide the LLM's output structure
        schema_json = RouterResponse.model_json_schema()
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        system_prompt = f"""
You are a Research Architect specializing in Gaming Intelligence. 
Today's Date: {current_date}.

STRICT VERSIONING RULES:
1. You DO NOT know the current patch version for {current_date} unless you have real-time proof.
2. If you are not 100% certain, you MUST set 'current_version' to "Unknown".
3. DO NOT hallucinate versions like 'v12.1' or '1.93'. Better to be "Unknown" than wrong.

SEARCH QUERY LOGIC:
- If 'current_version' is "Unknown", your FIRST query in 'official' MUST be to discover the version: 
  e.g., "latest patch version [Game Name] {current_date}"
- NEVER include old versions (like 'v12.1') in queries if the current state is "Unknown".
- Use terms like "latest", "actual meta", or "current update" instead.

SOURCE LOGIC:
- OFFICIAL: Developer blogs and technical update logs.
- PRO_STATISTICS: Dedicated analytic engines and high-level stat trackers.
- HIGH_TIER_COMMUNITY: Native language forums of the developers AND global hubs.

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
        
        # Validate the response against our Pydantic model
        raw_content = completion.choices[0].message.content
        return RouterResponse.model_validate_json(raw_content)

# Testing block for local development
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    agent = RouterAgent()
    try:
        test_plan = agent.generate_search_plan("Билд на друида в Аллодах Онлайн")
        print(test_plan.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error: {e}")