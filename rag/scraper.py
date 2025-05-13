from playwright.async_api import async_playwright
from pydantic import BaseModel
from openai import OpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from typing_extensions import List
import bs4

import os

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
        Gets the HTML content from the page as a string with a lot of white space because it adds many '\n'.
        """

        if partial:
            print('scraping_with_partil_true')
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
            
            # await self.page.goto(url, wait_until="load")
            # Convert URL to string to handle Pydantic HttpUrl objects
            url_str = str(url)
            await self.page.goto(url_str, wait_until="load")
            await self.page.wait_for_timeout(2000)  # Wait for dynamic content
            return await self.page.content()

    async def take_screenshot(self, path="screenshot.png"):
        await self.page.screenshot(path=path, full_page=True)
        return path
    async def screenshot_buffer(self):
        screenshot_bytes = await self.page.screenshot(type="png", full_page=False)
        return screenshot_bytes

    async def close(self):
        await self.browser.close()
        await self.playwright.stop()
        self.playwright = None
        self.browser = None
        self.page = None

# Scrape HTML content and 
class WebPageContent(BaseModel):
    mainUrl: str
    title: str
    description: str
    content: str

class WebPageContentList(BaseModel):
    pages: list[WebPageContent]

async def process_with_llm(html, instructions, truncate = False):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    """
    Process HTML content using an LLM to extract structured information.
    
    Args:
        html (str): The HTML content to process
        instructions (str): Specific instructions for the LLM on what to extract
        general_prompt (str): General context and guidelines for extraction
        truncate (bool, optional): Whether to truncate the HTML content. Defaults to False.
        
    Returns:
        WebPageContentList: Structured data extracted from the HTML content
    """
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[{
            "role": "system",
            "content": f"""
            You are an expert web scraping agent. Your task is to:
            Extract relevant information from this HTML to JSON 
            following these instructions:
            {instructions}

            Return ONLY valid JSON, no markdown or extra text."""
        }, {
            "role": "user",
            "content": html[:150000]  # Truncate to stay under token limits
        }],
        temperature=0.1,
        response_format=WebPageContentList,
        )
    return completion.choices[0].message.parsed