from playwright.async_api import async_playwright
from pydantic import BaseModel
from openai import OpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from typing_extensions import List
import bs4

class WebScraperAgent:
    """
    Inits a browser session. Can get content and take screenshots.
    """

    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page = None

    async def init_browser(self):
        # https://playwright.dev/python/docs/intro
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-accelerated-2d-canvas",
                "--disable-gpu",
                "--no-zygote",
                "--disable-audio-output",
                "--disable-software-rasterizer",
                "--disable-webgl",
                "--disable-web-security",
                "--disable-features=LazyFrameLoading",
                "--disable-features=IsolateOrigins",
                "--disable-background-networking"
            ]
        )
        self.page = await self.browser.new_page()

    async def scrape_content(self, url, partial=True) -> List[Document]:
        """
        Gets the HTML content from the page as a string. The final string has many white space because it adds many '\n'. The partial algorithm uses WebBaseLoader, while the integral algo uses Playwright to grab the entire HTML.
        Params
            URL (str): the target URL to scrape content from
            partial (boolean = True): whether to get content from only a few specific HTML tags
        
        Returns:
            List[Document]: List containing one or more Document objects with the page content
        """

        if partial:
            self.loader = WebBaseLoader(
                web_paths=(str(url),),
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        class_=("post-content", "post-title", "post-header")
                    )
                ),
            )
            
            docs = self.loader.load()
            return docs
        
        else:
            if not self.page or self.page.is_closed():
                await self.init_browser()
            
            # Convert URL to string to handle Pydantic HttpUrl objects
            url_str = str(url)
            await self.page.goto(url_str, wait_until="load")
            await self.page.wait_for_timeout(2000)  # Wait for dynamic content
            content = await self.page.content()
            # Convert the raw HTML string to a Document object
            return [Document(page_content=content)]

    async def close(self):
        await self.browser.close()
        await self.playwright.stop()
        self.playwright = None
        self.browser = None
        self.page = None