import os
import uuid
import json
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
    description="Professional Multi-Agent Engine with SSE Streaming and Memory.",
    version="2.1.0", 
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

class ChatMessage(BaseModel):
    session_id: str
    message: str

# Helper function to format SSE messages
def sse_message(event_type: str, data: dict) -> str:
    """Formats a message according to the Server-Sent Events spec."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

# 5. Routes
@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "API is running. Use /api/v1/deep-research for SSE streams."
    }

@app.get("/api/v1/deep-research")
async def deep_research(
    query: str = Query(..., min_length=3, description="The gaming query to research")
):
    """
    STREAMING ENDPOINT (SSE).
    Executes the pipeline and yields progress events to the frontend.
    """
    
    async def event_generator():
        try:
            print(f"\n[*] New stream started for query: {query}")
            
            # --- PHASE 1: ROUTING ---
            yield sse_message("status", {"step": "routing", "message": "Анализирую запрос и определяю игру..."})
            await asyncio.sleep(0.05)
            
            plan = await asyncio.to_thread(router_agent.generate_search_plan, query)
            
            yield sse_message("status", {
                "step": "routing_complete", 
                "message": f"Игра распознана: {plan.game_name}. Создаю маршрут поиска."
            })
            await asyncio.sleep(0.05)

            # --- PHASE 2: SCRAPING (Теперь слушаем каждый шаг!) ---
            search_results = []
            
            # Мы используем async for, чтобы получать логи от run_stream прямо в процессе работы
            async for event in scraper_agent.run_stream(plan.search_plan.model_dump()):
                if event["status"] == "done":
                    search_results = event["data"] # Это финальный результат Фазы 2
                else:
                    # Это промежуточные логи (поиск, нашел ссылки, скачал сайт 1 из 12...)
                    yield sse_message("status", {
                        "step": f"scraper_{event['status']}", 
                        "message": event["message"]
                    })
                    await asyncio.sleep(0.05)
            
            if not search_results:
                yield sse_message("error", {"message": "Не удалось найти релевантные данные в интернете."})
                return

            yield sse_message("status", {
                "step": "scraping_complete", 
                "message": f"Сбор данных завершен. Успешно прочитано сайтов: {len(search_results)}."
            })
            await asyncio.sleep(0.05)

            # --- PHASE 3: SYNTHESIS ---
            yield sse_message("status", {"step": "synthesizing", "message": "Синтезирую финальный гайд (Llama 70B)..."})
            await asyncio.sleep(0.05)
            
            final_output = await asyncio.to_thread(
                synthesizer_agent.generate_guide,
                query, plan.query_language, search_results
            )
            
            # --- SESSION CREATION ---
            session_id = str(uuid.uuid4())
            session_store[session_id] = {
                "context": search_results,
                "history": [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": final_output.markdown_guide}
                ]
            }
            
            print(f"[*] Stream complete. Session: {session_id}")
            
            # --- FINAL OUTPUT ---
            yield sse_message("complete", {
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
                "sources": [{"url": res['url']} for res in search_results]
            })

        except Exception as e:
            print(f"[CRITICAL ERROR] Pipeline failed: {str(e)}")
            yield sse_message("error", {"message": f"Произошла ошибка в пайплайне: {str(e)}"})

    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/api/v1/chat")
async def continue_chat(chat_req: ChatMessage):
    """Answers follow-up questions using cached session data."""
    if chat_req.session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session expired or not found.")
    
    session = session_store[chat_req.session_id]
    
    try:
        new_answer = await asyncio.to_thread(
            synthesizer_agent.continue_chat,
            chat_req.message,
            session["history"],
            session["context"]
        )
        
        session["history"].append({"role": "user", "content": chat_req.message})
        session["history"].append({"role": "assistant", "content": new_answer})
        
        return {"reply": new_answer}
        
    except Exception as e:
        print(f"[ERROR] Chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process follow-up question.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)