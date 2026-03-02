from typing import List, Dict, Any
from litellm import completion
from hybrid_search import search

SYSTEM_PROMPT = '''
You are a documentation search assistant. Your role is to help users find accurate documentation for libraries and frameworks.

## Your Task

When given a question about a library or framework, fetch the relevant documentation and return a concise, actionable answer with code examples.

## Process

**Return a focused answer**: Summarize the relevant documentation with:
   - Direct answer to the question
   - Code examples from the docs
   - Links or references if available
'''

def build_context(results: List[Dict[str, Any]], max_context_length: int = 4000) -> str:
    """
    Build a structured text context string from a list of search results.
    
    Args:
        results (List[Dict[str, Any]]): A list of dictionaries representing search results.
        max_context_length (int, optional): Maximum character length for the combined context. Defaults to 4000.
        
    Returns:
        str: A continuous string of relevant context aggregated from the search results.
    """
    context = []
    context_length = 0
    added_chunks = set()
  
    for result in results:
        chunk_type = result['type']
        section = result['content']
    
        if context_length + len(section) > max_context_length:
            break
    
        formatted = f"[{chunk_type.upper()}]\n{section}"
        context.append(formatted)
        context_length += len(formatted)
        added_chunks.add(result['id'])
  
    for result in results:
        if context_length >= max_context_length:
            break
    
        for related_chunk in result.get('related_chunks', []):
            if related_chunk['id'] in added_chunks:
                continue
    
            chunk_type = related_chunk['type']
            section = related_chunk['content']
    
            if context_length + len(section) > max_context_length:
                break
  
        formatted = f"[{chunk_type.upper()}]\n{section}"
        context.append(formatted)
        context_length += len(formatted)
        added_chunks.add(related_chunk['id'])

    return '\n\n'.join(context)

def generate_prompt(query: str, context: str) -> str:
    """
    Generate a prompt to format the user query alongside the contextual documentation.
    
    Args:
        query (str): The user's question.
        context (str): The retrieved documentation context to formulate the answer.
        
    Returns:
        str: A fully formatted prompt string.
    """
    prompt = f"""
    Documentation Context:
    {context}
  
    User Question:
    {query}
  
    Instructions:
    - Base your answer strictly on the provided documentation
    - If the documentation doesn't contain enough information, say so
    - Quote or reference specific code examples when relevant
    - Be clear and concise
  
    Answer:"""

    return prompt

def generate_response(query: str, explain_collection: Any, code_collection: Any, n_results: int = 10) -> Any:
    """
    Generate an answer to a query based on a hybrid search through documentation collections.
    
    Args:
        query (str): The user's question or search query.
        explain_collection (Any): The vector database collection containing explanatory chunks.
        code_collection (Any): The vector database collection containing code snippet chunks.
        n_results (int, optional): The number of search results to retrieve. Defaults to 10.
        
    Returns:
        Any: The generation response object from the language model.
    """
    results = search(query, explain_collection, code_collection, include_related=True, n_results=n_results)
    context = build_context(results)
    prompt = generate_prompt(query, context)
    response = completion(model="gemini/gemini-2.5-flash", messages=[
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": prompt}
    ])
    return response

