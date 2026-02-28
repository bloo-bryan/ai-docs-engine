import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from crawl4ai import AsyncWebCrawler, DefaultMarkdownGenerator, BrowserConfig, CrawlerRunConfig, MemoryAdaptiveDispatcher, CacheMode
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_filter_strategy import PruningContentFilter

@dataclass
class DocumentChunk:
    """
    Represents a chunk of documentation.

    Attributes:
        id (str): The unique identifier for the document chunk.
        content (str): The raw text content of the chunk.
        embedding_content (str): The text content formatted for generating embeddings.
        metadata (Dict[str, Any]): Additional metadata associated with the document.
    """
    id: str
    content: str
    embedding_content: str
    metadata: Dict[str, Any]

    def to_dict(self):
        """
        Converts the DocumentChunk instance to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the document chunk.
        """
        return asdict(self)

def parse_sitemap(sitemap: str) -> List[str]:
    """
    Fetches and parses an XML sitemap to extract all URL locations.

    Args:
        sitemap (str): The URL of the XML sitemap.

    Returns:
        list: A list of extracted URL strings. Returns an empty list if parsing fails.
    """
    print(f"Fetching sitemap from: {sitemap}")
    response = requests.get(sitemap)
    namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    urls = []

    if response.status_code == 200:
        print("Successfully fetched sitemap. Parsing XML...")
        root = ET.fromstring(response.content)
        try:
            # Find all <url> tags within the namespace
            for url in root.findall('ns:url', namespace):
                loc = url.find('ns:loc', namespace)
                if loc is not None:
                    urls.append(loc.text)
        except Exception as e:
            print(f"Error parsing sitemap XML: {e}")
    
    print(f"Extraction complete. Found {len(urls)} URLs in sitemap.")
    return urls

async def crawl_sitemap(urls: List[str]) -> Dict[str, ]:
    """
    Asynchronously crawls a list of URLs and extracts their main content into Markdown.

    Uses a memory-adaptive dispatcher and a pruning content filter to efficiently 
    extract meaningful content. Successfully crawled pages are saved as Markdown files 
    in a local directory.

    Args:
        urls (list): A list of URLs to crawl.

    Returns:
        dict: A dictionary mapping the crawled URL to its result object.
    """
    print(f"Preparing to crawl {len(urls)} URLs...")
    prune_filter = PruningContentFilter(
        threshold=0.5,
        threshold_type="dynamic"
    )
    crawl_config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=prune_filter,
            options={"ignore_links": True,
                     "skip_internal_links": True}   # ignore anchor links
        ),
        cache_mode=CacheMode.BYPASS,
        stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=10
    )

    async with AsyncWebCrawler() as crawler:
        print("Starting batch crawl with AsyncWebCrawler. This may take a while...")
        results = await crawler.arun_many(urls=urls, config=crawl_config, dispatcher=dispatcher)
        print("Batch crawl finished. Processing and saving results...")
        result_dict = dict()
        for result in results:
            if result.success:
                page_title = result.metadata.get("title", "Untitled").replace(":", " -")
                clean_title = re.sub(r'[\\/*?:"<>|]', " ", page_title)
                clean_title = re.sub(r'[^\x00-\x7F]+', '', clean_title)

                if not os.path.exists(DOC_DIR_NAME):
                    os.makedirs(DOC_DIR_NAME)

                with open(f"{DOC_DIR_NAME}/{clean_title}.md", "w", encoding="utf-8") as f:
                    f.write(result.markdown.fit_markdown)

                print(f"Successfully saved: {DOC_DIR_NAME}/{clean_title}.md")
                result_dict[result.url] = result

            else:
                print(f"Failed to crawl {result.url}: {result.error_message}")
        return result_dict