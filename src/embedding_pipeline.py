import re
import emoji
import chromadb
import asyncio
from litellm import acompletion
from chromadb.config import Settings
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from crawl4ai.models import CrawlResult
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import MarkdownHeaderTextSplitter

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

def process_split(chunk: Any) -> Tuple[str, List[str]]:
    """
    Extracts code blocks and text explanations from a given markdown split chunk.

    Args:
        chunk (Any): A markdown split object containing page_content.

    Returns:
        Tuple[str, List[str]]: A tuple containing the text explanation without code blocks, and a list of extracted code blocks.
    """
    code_pattern = r"```(.+)```"
    replace_pattern = r"```[\s\S]*?```"
    codes = re.findall(code_pattern, chunk.page_content, re.DOTALL)
    explanation = re.sub(replace_pattern, '', chunk.page_content, re.DOTALL)
    return (explanation, codes)

def id_generator(title: str, code: bool = False) -> str:
    """
    Generates a unique identifier string based on the section title.

    Args:
        title (str): The section title.
        code (bool, optional): Whether the ID is for a code chunk. Defaults to False.

    Returns:
        str: The generated identifier string.
    """
    clean_title = emoji.replace_emoji(title, replace="").strip()
    if code:
        return f"{clean_title}_code"
    else:
        return f"{clean_title}_explanation"

async def enrich_chunk(chunk: str, result: CrawlResult = None) -> str:
    """
    Enriches a text or code chunk by using an LLM to generate contextual explanation relative to the document.

    Args:
        chunk (str): The chunk string to be enriched.
        result (CrawlResult, optional): The crawl result object providing the full document context. Defaults to None.

    Returns:
        str: The enriched chunk text including the context provided by the LLM.
    """
    prompt = f"""<document>
    Title: f"{result.metadata['title']}"
    
    {result.markdown.fit_markdown[:4000]}
    </document>

    <chunk>
    {chunk}
    </chunk>

    Provide a brief, 1-2 sentence context explaining what this chunk discusses in relation to the overall document.
    The context should help someone understand this chunk without seeing the full document.

    Format your response as:
    "This chunk from [document title] discusses [brief explanation]."

    Be concise and specific. Do not include any preamble or explanation, just the context sentence(s).
    """

    response = await acompletion(model="gemini/gemini-2.5-flash", messages=[{"role": "user", "content": prompt}])
    return f"{response.choices[0].message.content}\n\n{chunk}"

async def create_chunks(title: str, explanation: str, codes: List[str], result: CrawlResult) -> Tuple[List[DocumentChunk], List[DocumentChunk]]:
    """
    Creates DocumentChunk instances for explanations and code extracted from a section.

    Args:
        title (str): The title of the document section.
        explanation (str): The text explanation content.
        codes (List[str]): A list of code blocks.
        result (CrawlResult): The overall crawl result object for context.

    Returns:
        Tuple[List[DocumentChunk], List[DocumentChunk]]: A tuple of lists containing explanation chunks and code chunks.
    """
    explanation_chunk, code_chunks = [], []
    code_ids = [f"{id_generator(title, code=True)}_{i}" for i in range(len(codes))]
    
    tasks = []
    if explanation:
        tasks.append(enrich_chunk(explanation, result))
    if codes:
        tasks.extend([enrich_chunk(code, result) for code in codes])

    enriched_results = await asyncio.gather(*tasks)

    if explanation:
    #   enriched_explanation = await enrich_chunk(title, explanation, result)
      explanation_chunk.append(DocumentChunk(
          id = id_generator(title),
          content = explanation,
          embedding_content = enriched_results.pop(0),
          metadata = {
              'type': 'explanation',
              'source_url': result.url,
              'section_title': title,
              'has_code_examples': bool(codes),
              'related_ids': ','.join(code_ids)
          },
      ))
    if codes:
      for i, code in enumerate(codes):
        # enriched_code = await enrich_chunk(title, code, result)
        code_chunks.append(DocumentChunk(
            id = code_ids[i],
            content = code,
            embedding_content = enriched_results.pop(0),
            metadata = {
                'type': 'code',
                'source_url': result.url,
                'section_title': title,
                'related_ids': explanation_chunk[0].id if explanation_chunk else ''
            },
        ))
    return (explanation_chunk, code_chunks)

async def process_results(results: Dict[str, CrawlResult]) -> Tuple[List[DocumentChunk], List[DocumentChunk]]:
    """
    Processes a dictionary of crawl results into documentation chunks for explanation and code.

    Args:
        results (Dict[str, CrawlResult]): A dictionary mapping URLs to their corresponding CrawlResult objects.

    Returns:
        Tuple[List[DocumentChunk], List[DocumentChunk]]: A tuple containing a list of explanation chunks and a list of code chunks.
    """
    all_explanation_chunks, all_code_chunks = [], []
    chunk_tasks = []

    for url, result in results.items():
        # Define the headers to split on and their metadata keys
        headers = [
            ("#", "H1"),
            ("##", "H2"),
            ("###", "H3")
        ]
  
        # Initialize the splitter
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
  
        # Split the text
        splits = markdown_splitter.split_text(result.markdown.fit_markdown)
  
        for split in splits:
            section_title = "_".join(list(split.metadata.values())[-2:]).lower().strip()
            section_title = re.sub(r"[!@#$%^&*<>?\/|}{~:\s]", "_", section_title)
            explanation, codes = process_split(split)
            
            # Add the create_chunks coroutine to our task list instead of awaiting it sequentially
            chunk_tasks.append(
                create_chunks(section_title, explanation.strip(), codes, result)
            )
  
        # Execute all chunk creation tasks concurrently
        if chunk_tasks:
            gathered_results = await asyncio.gather(*chunk_tasks)
            for exp_chunks, code_chunks in gathered_results:
                if exp_chunks:
                    all_explanation_chunks.extend(exp_chunks)
                if code_chunks:
                    all_code_chunks.extend(code_chunks)
    
        return all_explanation_chunks, all_code_chunks

def create_embeddings(texts: List[str], code: bool = False) -> Any:
    """
    Creates vector embeddings for a given list of strings using the appropriate embedding model.

    Args:
        texts (List[str]): A list of text strings to embed.
        code (bool, optional): Whether to use the code-specific embedding model. Defaults to False.

    Returns:
        Any: The generated embeddings, typically as a numpy array or tensor.
    """
    model = code_model if code else text_model
    embeddings = model.encode(texts)
    print(f"Created embeddings for {len(texts)} texts, shape: {embeddings.shape}")
    return embeddings

def initialize_chroma(collection_name: str) -> Any:
    """
    Initializes a persistent ChromaDB client and creates or retrieves a collection by name.
    If the collection already exists, it will be deleted and recreated.

    Args:
        collection_name (str): The name of the collection to initialize.

    Returns:
        Any: The initialized ChromaDB collection object.
    """
    chroma = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)
    collection = chroma.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def add_documents_to_chroma(chunks: List[DocumentChunk], db_name: str, code: bool = False) -> Any:
    """
    Adds a list of document chunks and their generated embeddings to a specified ChromaDB collection.

    Args:
        chunks (List[DocumentChunk]): A list of DocumentChunk instances to add.
        db_name (str): The name of the database collection.
        code (bool, optional): Whether to use the code-specific embedding model for the collection. Defaults to False.

    Returns:
        Any: The ChromaDB collection object with the added documents.
    """
    collection = initialize_chroma(db_name)
    original, texts, metas, ids = [], [], [], []
    for chunk in chunks:
        original.append(chunk.content)
        texts.append(chunk.embedding_content)
        metas.append(chunk.metadata)
        ids.append(chunk.id)
    vectors = create_embeddings(texts, code)
    collection.add(
        documents=original,
        metadatas=metas,
        ids=ids,
        embeddings=vectors
    )
    print(f"Vectorstore created with {collection.count()} documents")
    return collection