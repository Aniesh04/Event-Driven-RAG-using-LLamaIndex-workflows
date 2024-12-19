import asyncio
# from ddg import Duckduckgo
from duckduckgo_search import DDGS
from crawl4ai import AsyncWebCrawler
from asyncio import ProactorEventLoop

asyncio.set_event_loop(ProactorEventLoop())

class WebSearch:

    def __init__(self,top_n_urls=5):

        self.top_n_urls=top_n_urls

        self.ddg_api = DDGS()

    def get_urls(self,searc_query):

        "Collects URLs using the duckduckgo API"
        
        results = self.ddg_api.text(searc_query,max_results=self.top_n_urls)

        urls = [i['href'] for i in results]

        return urls
    
    async def scrap_url(self,url):
        "Scraps Individual URLs"

        async with AsyncWebCrawler(verbose=True) as crawler:

            result = await crawler.arun(url=url)

            return (url,result.markdown)
        
    async def search_online(self,search_query):

        "Main method to scrap data from websites"

        urls = self.get_urls(search_query)

        # lst = [url['href'] for url in urls]
        scrap_text = []
        for url in urls:
          scrap_text.append(await self.scrap_url(url))

        documents = []
        for i in range(len(scrap_text)):
          documents.append(scrap_text[i][1])

        return documents
    
    # def run_search(self,search_query):
    # # Set ProactorEventLoop for Windows
    #     if asyncio.get_event_loop().is_running():
    #         asyncio.set_event_loop(asyncio.new_event_loop())

    #     loop = asyncio.ProactorEventLoop() if hasattr(asyncio, 'ProactorEventLoop') else asyncio.get_event_loop()
    #     asyncio.set_event_loop(loop)
    #     return loop.run_until_complete(self.search_online(search_query))
    async def run_search(self, search_query):
        return await self.search_online(search_query)

if __name__ == "__main__":

    search_Engine = WebSearch()

    # content = asyncio.run(search_Engine.run_search("Who is David Warner"))
    content = search_Engine.run_search("Who is David Warner")

    print("-----------------------------------")
    print(content)
    print("-----------------------------------")

