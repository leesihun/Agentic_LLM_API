"""
Web Search Tool Implementation
Pure Tavily search - no LLM processing
All result interpretation handled by ReAct Agent
"""
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

import config
from backend.utils.prompts_log_append import log_to_prompts_file


class WebSearchTool:
    """
    Web search using Tavily API
    Returns raw search results without any LLM processing
    """

    def __init__(self):
        """Initialize web search tool"""
        self.api_key = config.TAVILY_API_KEY
        self.max_results = config.WEBSEARCH_MAX_RESULTS
        self.search_depth = config.TAVILY_SEARCH_DEPTH
        self.include_domains = config.TAVILY_INCLUDE_DOMAINS
        self.exclude_domains = config.TAVILY_EXCLUDE_DOMAINS

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        search_depth: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform web search using Tavily (no LLM processing)

        Args:
            query: Raw search query from user/agent
            max_results: Override default max results
            search_depth: "basic" or "advanced"
            include_domains: List of domains to include
            exclude_domains: List of domains to exclude

        Returns:
            Dict with raw Tavily results
        """
        if getattr(config, "AIRGAPPED", True):
            # No outbound internet: fail fast instead of hanging on a connect timeout.
            return {
                "success": False,
                "error": "websearch is disabled: this deployment is airgapped (no internet access)",
                "hint": "Use local files (file_reader/grep/file_navigator), uploaded documents (rag), or ask the user.",
                "query": query,
                "results": [],
                "num_results": 0,
            }

        # Log to file
        log_to_prompts_file("\n\n")
        log_to_prompts_file("=" * 80)
        log_to_prompts_file(f"TOOL EXECUTION: websearch")
        log_to_prompts_file("=" * 80)
        log_to_prompts_file(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_to_prompts_file(f"INPUT:")
        log_to_prompts_file(f"  Query: {query}")
        log_to_prompts_file(f"  Max Results: {max_results or self.max_results}")
        log_to_prompts_file(f"  Search Depth: {search_depth or self.search_depth}")

        print("\n" + "=" * 80)
        print("[WEBSEARCH TOOL] search() called")
        print("=" * 80)
        print(f"Query: {query}")
        print(f"Max results: {max_results or self.max_results}")
        print(f"Search depth: {search_depth or self.search_depth}")

        try:
            from tavily import TavilyClient
        except ImportError:
            error_msg = "Tavily client not installed. Install with: pip install tavily-python"
            print(f"[WEBSEARCH] [ERROR] {error_msg}")
            raise ImportError(error_msg)

        # Use provided values or defaults
        max_res = max_results or self.max_results
        depth = search_depth or self.search_depth
        inc_domains = include_domains or self.include_domains
        exc_domains = exclude_domains or self.exclude_domains

        # Initialize client and perform search
        client = TavilyClient(api_key=self.api_key)
        start_time = time.time()

        search_params = {
            "query": query,
            "max_results": max_res,
            "search_depth": depth
        }

        if inc_domains:
            search_params["include_domains"] = inc_domains
        if exc_domains:
            search_params["exclude_domains"] = exc_domains

        print(f"[WEBSEARCH] Calling Tavily API...")
        results = client.search(**search_params)
        execution_time = time.time() - start_time

        num_results = len(results.get("results", []))
        print(f"[WEBSEARCH] Found {num_results} results in {execution_time:.2f}s")

        # Log results to file
        log_to_prompts_file(f"OUTPUT:")
        log_to_prompts_file(f"  Status: SUCCESS")
        log_to_prompts_file(f"  Results Found: {num_results}")
        log_to_prompts_file(f"  Execution Time: {execution_time:.2f}s")

        if num_results > 0:
            log_to_prompts_file(f"RESULTS:")
            for i, result in enumerate(results.get("results", []), 1):
                log_to_prompts_file(f"  [{i}] {result.get('title', 'N/A')}")
                log_to_prompts_file(f"      URL: {result.get('url', 'N/A')}")
                log_to_prompts_file(f"      Score: {result.get('score', 0.0):.3f}")
                log_to_prompts_file(f"      Content: {result.get('content', 'N/A')[:200]}...")

        log_to_prompts_file("=" * 80)

        return {
            "success": True,
            "results": results.get("results", []),
            "query": query,
            "execution_time": execution_time,
            "num_results": num_results
        }

