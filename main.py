import os
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Importing our internal modules
from app.agents.router import RouterAgent
from app.agents.scraper import ScraperAgent 
from app.agents.synthesizer import SynthesizerAgent

# 1. Configuration & Environment
load_dotenv()

# 2. Application Setup
app = FastAPI(
    title="Gamer Meta Oracle API",
    description="Professional Multi-Agent Engine with Conversational Memory.",
    version="1.1.0", # Version bump for Chat feature
    docs_url="/docs",
    redoc_url="/redoc"
)

# 3. Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Agent Initialization & Memory Store
router_agent = RouterAgent()
scraper_agent = ScraperAgent()
synthesizer_agent = SynthesizerAgent()

# IN-MEMORY STORE: Holds scraped data and chat history mapped to a session_id
session_store = {}

# Pydantic schema for the chat endpoint
class ChatMessage(BaseModel):
    session_id: str
    message: str

# 5. Routes
@app.get("/")
async def root():
    """Welcome endpoint for API health check."""
    return {
        "status": "online",
        "message": "Gamer Meta Oracle API is running. Head to /docs for usage."
    }

@app.get("/api/v1/search-plan")
async def get_search_plan(
    query: str = Query(..., min_length=3, description="The gaming query to analyze")
):
    """
    PHASE 1: Routing & Research Strategy.
    """
    try:
        plan = router_agent.generate_search_plan(query)
        return plan
    except Exception as e:
        print(f"[ERROR] Phase 1 failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Routing Agent failed.")

@app.get("/api/v1/deep-research")
async def deep_research(
    query: str = Query(..., min_length=3, description="The gaming query to research")
):
    """
    PHASE 1, 2 & 3: Full Research Pipeline.
    Initializes a chat session upon completion.
    """
    try:
        print(f"[*] Starting Phase 1 (Routing) for query: '{query}'")
        plan = router_agent.generate_search_plan(query)
        
        print(f"[*] Starting Phase 2 (Scraping) for {plan.game_name}...")
        search_results = await scraper_agent.run(plan.search_plan.model_dump())
        
        if not search_results:
            raise ValueError("Scraper could not find any relevant data.")

        print(f"[*] Starting Phase 3 (Synthesis)... Analyzing {len(search_results)} sources.")
        final_output = synthesizer_agent.generate_guide(
            user_query=query,
            query_language=plan.query_language,
            scraped_data=search_results
        )
        
        # --- NEW: SESSION CREATION ---
        session_id = str(uuid.uuid4())
        session_store[session_id] = {
            "context": search_results, # Save the raw data so we don't have to scrape again
            "history": [
                {"role": "user", "content": query},
                {"role": "assistant", "content": final_output.markdown_guide}
            ]
        }
        
        print(f"[*] Pipeline complete! Session ID generated: {session_id}")
        
        return {
            "session_id": session_id,
            "query": query,
            "game_info": {
                "name": plan.game_name,
                "origin": plan.context.game_origin,
                "detected_language": plan.query_language,
                "verified_version": final_output.verified_version
            },
            "meta_analysis": {
                "confidence_score": final_output.confidence_score,
                "guide": final_output.markdown_guide
            },
            "sources": [res['url'] for res in search_results]
        }
        
    except Exception as e:
        print(f"[CRITICAL ERROR] Pipeline failed: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Research pipeline failed: {str(e)}"
        )

@app.post("/api/v1/chat")
async def continue_chat(chat_req: ChatMessage):
    """
    PHASE 4: Chat Continuation.
    Answers follow-up questions instantly using cached session data.
    """
    if chat_req.session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session expired or not found. Please start a new deep-research.")
    
    session = session_store[chat_req.session_id]
    
    try:
        print(f"[*] Processing follow-up for session {chat_req.session_id[:8]}...")
        # Get answer from Synthesizer using saved context and history
        new_answer = synthesizer_agent.continue_chat(
            new_message=chat_req.message,
            chat_history=session["history"],
            scraped_data=session["context"]
        )
        
        # Update memory with the new exchange
        session["history"].append({"role": "user", "content": chat_req.message})
        session["history"].append({"role": "assistant", "content": new_answer})
        
        return {"reply": new_answer}
        
    except Exception as e:
        print(f"[ERROR] Chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process follow-up question.")

# 6. Server Execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)