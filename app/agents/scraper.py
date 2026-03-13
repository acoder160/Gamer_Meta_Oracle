import asyncio
import datetime
from typing import List, Dict
from ddgs import DDGS
from trafilatura import fetch_url, extract

class ScraperAgent:
    """
    Phase 2 Agent: Responsible for executing search queries 
    and extracting clean text content from web pages.
    Optimized for Deep Research (parses up to 12 sources concurrently).
    """
    
    def __init__(self):
        self.ddgs = DDGS()

    # УВЕЛИЧИЛИ max_results до 3, чтобы собирать больше ссылок на каждый запрос
    async def get_urls(self, queries: List[str], max_results: int = 3) -> List[str]:
        urls = []
        current_year = datetime.datetime.now().year
        # Создаем список "запрещенных" старых годов (например: 2024, 2023, 2022, 2021)
        forbidden_years = [str(current_year - i) for i in range(2, 6)] 
        
        for query in queries:
            try:
                results = await asyncio.to_thread(self.ddgs.text, query, max_results=max_results, timelimit='y')
                if results:
                    for r in results:
                        url = r['href']
                        # ANTI-SEO FILTER: Если в ссылке есть старый год - выкидываем её!
                        if any(old_year in url for old_year in forbidden_years):
                            print(f"[-] Blocked SEO Trap URL: {url}")
                            continue
                        urls.append(url)
            except Exception as e:
                print(f"Error searching for '{query}': {e}")
        
        return list(set(urls))

    async def scrape_page(self, url: str) -> Dict[str, str]:
        """
        Downloads a page and extracts clean content.
        """
        try:
            downloaded = await asyncio.to_thread(fetch_url, url)
            if downloaded:
                content = await asyncio.to_thread(extract, downloaded, include_comments=False, include_tables=True)
                if content:
                    return {"url": url, "content": content[:6000]} 
        except Exception as e:
            print(f"Error scraping {url}: {e}")
        return None

    async def run_stream(self, search_plan: Dict[str, List[str]]):
        """
        Streaming execution loop. Yields progress events back to the main API.
        """
        all_queries = []
        for category in search_plan.values():
            if isinstance(category, list):
                all_queries.extend(category)

        yield {"status": "searching", "message": f"Выполняю глубокий поиск по {len(all_queries)} запросам..."}
        
        urls = await self.get_urls(all_queries)
        
        if not urls:
            yield {"status": "done", "data": []}
            return

        # УВЕЛИЧИЛИ ЛИМИТ ДО 12 ССЫЛОК (10+-)
        urls_to_scrape = urls[:12] 
        
        yield {
            "status": "links_found", 
            "message": f"Найдена база из {len(urls)} ссылок. Начинаю чтение {len(urls_to_scrape)} лучших источников..."
        }

        results = []
        tasks = [self.scrape_page(url) for url in urls_to_scrape]
        
        completed_count = 0
        
        for task in asyncio.as_completed(tasks):
            res = await task
            completed_count += 1
            
            if res:
                results.append(res)
                yield {
                    "status": "reading", 
                    "message": f"[{completed_count}/{len(urls_to_scrape)}] Успешно прочитано: {res['url'][:50]}..."
                }
            else:
                yield {
                    "status": "reading_error", 
                    "message": f"[{completed_count}/{len(urls_to_scrape)}] Заблокирован доступ к сайту."
                }

        yield {"status": "done", "data": results}