import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Importing our internal modules
from app.agents.router import RouterAgent
from app.agents.scraper import ScraperAgent 

# 1. Configuration & Environment
load_dotenv()

# 2. Application Setup
app = FastAPI(
    title="Gamer Meta Oracle API",
    description="Professional Multi-Agent Engine for gaming meta-research and analysis.",
    version="0.3.0",
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

# 4. Agent Initialization
router_agent = RouterAgent()
scraper_agent = ScraperAgent()

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
    Returns the structured search plan without executing it.
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
    PHASE 1 & 2: Full Research Pipeline.
    1. RouterAgent creates a search plan.
    2. ScraperAgent executes the plan and extracts raw data from the web.
    """
    try:
        # Step 1: Generate the plan
        print(f"[*] Starting Phase 1 for query: {query}")
        plan = router_agent.generate_search_plan(query)
        
        # Step 2: Execute scraping based on the plan
        # We convert the Pydantic model to a dict for the Scraper
        print(f"[*] Starting Phase 2: Scraping for {plan.game_name}")
        search_results = await scraper_agent.run(plan.search_plan.model_dump())
        
        return {
            "query": query,
            "game_info": {
                "name": plan.game_name,
                "origin": plan.context.game_origin,
                "detected_language": plan.query_language
            },
            "research_results": search_results,
            "sources_count": len(search_results)
        }
        
    except Exception as e:
        print(f"[CRITICAL ERROR] Pipeline failed: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Research pipeline failed: {str(e)}"
        )

# 6. Server Execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)