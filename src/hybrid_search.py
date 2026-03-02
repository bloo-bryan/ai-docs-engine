from dataclasses import dataclass
import re
import numpy as np
from rank_bm25 import BM25Okapi
from collections import defaultdict
from typing import Dict, List, Any, Tuple

@dataclass
class SearchResult:
    doc_id: str
    score: float
    content: str
    metadata: dict
    source: str

def tokenize_text(text: str) -> List[str]:
    """
    Tokenize standard text by splitting on whitespace and converting to lowercase.
    
    Args:
        text (str): The input string to text.
        
    Returns:
        List[str]: List of lowercase tokens.
    """
    return text.lower().split()

def tokenize_code(code: str) -> List[str]:
    """
    Tokenize code snippets by extracting words, method calls, and common syntax characters.
    
    Args:
        code (str): The input code snippet string.
        
    Returns:
        List[str]: List of lowercase code tokens.
    """
    pattern = r'\w+\.[\w.]+|\w+|[(){}\[\]]'
    tokens = re.findall(pattern, code.lower())
    return tokens

def build_indices(chunks: List[SearchResult]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build BM25 indices from a collection of SearchResult chunks.
    
    Args:
        chunks (List[SearchResult]): List of SearchResult objects containing explanatory and code snippets.
        
    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: A tuple of two dictionaries representing the BM25
        indices for explanations and code, respectively.
    """
    bm25_explanations, bm25_codes = defaultdict(list), defaultdict(list)
    for chunk in chunks:
        if chunk.metadata['type'] == 'explanation':
            bm25_explanations['doc_ids'].append(chunk.doc_id)
            bm25_explanations['content'].append(tokenize_text(chunk.content))
        if chunk.metadata['type'] == 'code':
            bm25_codes['doc_ids'].append(chunk.doc_id)
            bm25_codes['content'].append(tokenize_code(chunk.content))

    bm25_explanations['content'] = BM25Okapi(bm25_explanations['content'])
    bm25_codes['content'] = BM25Okapi(bm25_codes['content'])

    return (bm25_explanations, bm25_codes)


def search_semantic(query: List[float], collection: Any, n_results: int = 10) -> List[SearchResult]:
    """
    Perform a semantic search on the collection using query embeddings.
    
    Args:
        query (List[float]): The query embedding vector.
        collection (Any): The vector database collection to query.
        n_results (int, optional): Maximum number of results to return. Defaults to 10.
        
    Returns:
        List[SearchResult]: List of SearchResult objects matching the query.
    """
    results = collection.query(
        query_embeddings=[query],
        n_results=n_results
    )

    return [
        SearchResult(
            doc_id = results['ids'][0][i],
            score = results['distances'][0][i],
            content = results['documents'][0][i],
            metadata = results['metadatas'][0][i],
            source = 'semantic'
        )
        for i in range(n_results)
    ]

def search_bm25(query: str, index: Dict[str, Any], collection: Any, n_results: int = 10) -> List[SearchResult]:
    """
    Perform a keyword-based search using the BM25 Okapi algorithm.
    
    Args:
        query (str): The raw string query.
        index (Dict[str, Any]): The BM25 index dictionary constructed from build_indices.
        collection (Any): The vector database collection to fetch full representations from.
        n_results (int, optional): Maximum number of results to return. Defaults to 10.
        
    Returns:
        List[SearchResult]: List of SearchResult objects retrieved based on BM25 scores.
    """
    tokenized_query = tokenize_text(query)
    scores = index['content'].get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:n_results]
    top_idx_scores = [
        (index['doc_ids'][idx], float(scores[idx]))
        for idx in top_indices if scores[idx] > 0
    ]

    results = []
    for doc_id, score in top_idx_scores:
        try:
            data = collection.get(
                ids=[doc_id],
                include=['documents','metadatas']
            )
    
            if data['ids']:
              results.append(SearchResult(
                  doc_id=doc_id,
                  score=score,
                  content=data['documents'][0],
                  metadata=data['metadatas'][0],
                  source='bm25'
              ))
        except Exception as e:
            print(f"Error retrieving data for doc_id {doc_id}: {e}")
    return results

def reciprocal_rank_fusion(sorted_lists: List[List[SearchResult]], k: int = 60) -> List[SearchResult]:
    """
    Combine results from multiple search strategies using Reciprocal Rank Fusion.
    
    Args:
        sorted_lists (List[List[SearchResult]]): A list containing individual lists of SearchResult objects.
        k (int, optional): The constant added to each rank to smooth inverse rank scores. Defaults to 60.
        
    Returns:
        List[SearchResult]: List of SearchResult objects re-ranked based on fusion scores.
    """
    rrf_scores, doc_map = defaultdict(float), dict()
    for sorted_list in sorted_lists:
        for rank, result in enumerate(sorted_list):
            doc_id = result.doc_id
            rrf_scores[doc_id] += 1.0 / (k + rank)
            doc_map[doc_id] = result

    sorted_rrf_scores = sorted(rrf_scores.items(), key=lambda x: -x[1])

    results = []
    for doc_id, rrf_score in sorted_rrf_scores:
        result = doc_map[doc_id]
        results.append(SearchResult(
            doc_id=result.doc_id,
            score=rrf_score,
            content=result.content,
            metadata=result.metadata,
            source='hybrid'
        ))
    return results

def fetch_related_chunks(collection: Any, related_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch related dataset chunks from the given collection based on document IDs.
    
    Args:
        collection (Any): The vector database collection containing chunks.
        related_ids (List[str]): List of chunk ID strings to fetch.
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing fetched chunk data.
    """
    data = collection.get(
        ids=related_ids,
        include=['documents','metadatas']
    )
    related_chunks = []
    for i in range(len(data['ids'])):
        related_chunks.append({
            'id': data['ids'][i],
            'content': data['documents'][i],
            'metadata': data['metadatas'][i],
            'type': data['metadatas'][i]['type']
        })
    return related_chunks

def search(query: str, explain_collection: Any, code_collection: Any, include_related: bool = True, n_results: int = 10) -> List[Dict[str, Any]]:
    """
    Execute a full hybrid search aggregating semantic and keyword strategies across explanation and code collections.
    
    Args:
        query (str): The search query string.
        explain_collection (Any): The collection containing explanatory chunks.
        code_collection (Any): The collection containing code snippet chunks.
        include_related (bool, optional): Whether to fetch and embed logically related chunks. Defaults to True.
        n_results (int, optional): Maximum number of merged results to return. Defaults to 10.
        
    Returns:
        List[Dict[str, Any]]: List of combined, ranked result dictionaries with associated raw content and related chunks.
    """
    # Note: create_embeddings must be available in the execution scope.
    embedded_query = create_embeddings(query)
  
    explain_results_semantic = search_semantic(embedded_query, explain_collection, n_results)
    code_results_semantic = search_semantic(embedded_query, code_collection, n_results)
    all_semantic_results = explain_results_semantic + code_results_semantic
    semantic_results_sorted = sorted(all_semantic_results, key=lambda x: x.score)
  
    bm25_explanations, bm25_codes = build_indices(all_semantic_results)
  
    explain_results_bm25 = search_bm25(query, bm25_explanations, explain_collection, n_results)
    code_results_bm25 = search_bm25(query, bm25_codes, code_collection, n_results)
    all_bm25_results = explain_results_bm25 + code_results_bm25
    bm25_results_sorted = sorted(all_bm25_results, key=lambda x: -x.score)
  
    fused_results = reciprocal_rank_fusion([semantic_results_sorted, bm25_results_sorted])
    top_fused_results = fused_results[:n_results]
  
    final_results = []
    for result in top_fused_results:
        result_dict = {
            'id': result.doc_id,
            'content': result.content,
            'metadata': result.metadata,
            'type': result.metadata['type'],
            'score': result.score,
            'source': result.source,
            'related_chunks': []
        }
        if include_related:
            related_ids = result.metadata['related_ids'].split(',')
            if related_ids:
                is_code = result.metadata['type'] == 'explanation'
                result_dict['related_chunks'] = fetch_related_chunks(code_collection if is_code else explain_collection, related_ids)

        final_results.append(result_dict)
    return final_results

