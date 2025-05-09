from playwright.async_api import async_playwright
from pydantic import BaseModel
import os
from openai import OpenAI

class WebScraperAgent:
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

    async def scrape_content(self, url):
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

scraper = WebScraperAgent()


class WebPageContent(BaseModel):
    title: str
    description: str
    domain: str
    imageUrl: str
    mainUrl: str


class WebPageContentList(BaseModel):
    pages: list[WebPageContent]


client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
async def process_with_llm(html, instructions, general_prompt, truncate = False):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[{
            "role": "system",
            "content": f"""
            You are an expert web scraping agent. Your task is to:
            Extract relevant information from this HTML to JSON 
            following these instructions:
            {instructions}
            
            {general_prompt}

            Return ONLY valid JSON, no markdown or extra text."""
        }, {
            "role": "user",
            "content": html[:150000]  # Truncate to stay under token limits
        }],
        temperature=0.1,
        response_format=WebPageContentList,
        )
    return completion.choices[0].message.parsed


async def webscraper(target_url, instructions):
    result = None
    try:
        # Ensure URL is a string
        target_url_str = str(target_url)
        
        # Scrape content and capture screenshot
        print("Extracting HTML Content \n")
        html_content = await scraper.scrape_content(target_url_str)

        print("Taking Screenshot \n")
        screenshot = await scraper.screenshot_buffer()
        # Process content

        print("Processing..")
        general_prompt = """
            Extract the title, description, presenter, the image URL and course URL for each of all the courses for the deeplearning.ai website.
            Add a scrapped_at parameter with the datetime of the operation.
        """
        result: WebPageContentList = await process_with_llm(html_content, instructions, general_prompt, False)
        print("\nGenerated Structured Response")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        await scraper.close()
    return result, screenshot