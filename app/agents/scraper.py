import asyncio
from typing import List, Dict
from ddgs import DDGS
from trafilatura import fetch_url, extract

class ScraperAgent:
    """
    Phase 2 Agent: Responsible for executing search queries 
    and extracting clean text content from web pages.
    """
    
    def __init__(self):
        # We use DDGS as it's free, fast, and doesn't require API keys
        self.ddgs = DDGS()

    async def get_urls(self, queries: List[str], max_results: int = 2) -> List[str]:
        """
        Executes a list of search queries and returns a deduplicated list of URLs.
        """
        urls = []
        for query in queries:
            try:
                # Running the search
                results = self.ddgs.text(query, max_results=max_results)
                if results:
                    urls.extend([r['href'] for r in results])
            except Exception as e:
                print(f"Error searching for '{query}': {e}")
        
        # Return unique URLs only
        return list(set(urls))

    async def scrape_page(self, url: str) -> Dict[str, str]:
        """
        Downloads a page and extracts clean content.
        """
        try:
            # fetch_url downloads the HTML
            downloaded = fetch_url(url)
            if downloaded:
                # extract() converts HTML to clean Markdown-like text
                content = extract(downloaded, include_comments=False, include_tables=True)
                if content:
                    return {"url": url, "content": content[:5000]} # Limit text length per page
        except Exception as e:
            print(f"Error scraping {url}: {e}")
        return None

    async def run(self, search_plan: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """
        The main workflow:
        1. Flatten all queries from the plan (official, pro, community).
        2. Get all URLs.
        3. Scrape and return the data.
        """
        all_queries = []
        for category in search_plan.values():
            all_queries.extend(category)

        # Step 1: Get URLs
        print(f"[*] Executing {len(all_queries)} search queries...")
        urls = await self.get_urls(all_queries)
        
        if not urls:
            return []

        # Step 2: Scrape content concurrently to be fast
        print(f"[*] Found {len(urls)} links. Starting extraction...")
        tasks = [self.scrape_page(url) for url in urls[:6]] # Limit to top 6 sources for speed/tokens
        results = await asyncio.gather(*tasks)
        
        # Return only successful results
        return [r for r in results if r is not None]