import json
import os
from typing import List, Dict, Any, Optional, Callable
from openai import OpenAI
from pydantic import BaseModel, Field
from vector_search_tool import VectorSearchTool
from grep_tool import GrepTool
import time


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None


class RAGAgent:
    """Simple RAG agent with o3 model and tools"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "o3"):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.vector_tool = VectorSearchTool(api_key=api_key)
        self.grep_tool = GrepTool()
        
        # Available models in order of preference
        self.available_models = [
            "o3",
            "o1", 
            "o1-preview",
            "gpt-4o",
            "gpt-4o-mini"
        ]
        
        self.active_model = self._get_available_model()
        print(f"Using model: {self.active_model}")
    
    def _get_available_model(self) -> str:
        """Check which models are available and return the best one"""
        for model in self.available_models:
            try:
                # Test with a simple completion
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                print(f"Successfully tested model: {model}")
                return model
            except Exception as e:
                print(f"Model {model} not available: {e}")
                continue
        
        # If none work, raise error
        raise ValueError("No compatible models available")
    
    def load_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """Load documents into both vector and grep tools"""
        print("Loading documents into RAG system...")
        
        # Load into vector search
        self.vector_tool.build_index(documents, metadata)
        
        # Load into grep tool
        self.grep_tool.load_documents(documents, metadata)
        
        print(f"Loaded {len(documents)} documents into RAG system")
    
    def vector_search(self, query: str, top_k: int = 5) -> ToolResult:
        """Execute vector similarity search"""
        try:
            results = self.vector_tool.search(query, top_k)
            return ToolResult(
                tool_name="vector_search",
                success=True,
                result=results
            )
        except Exception as e:
            return ToolResult(
                tool_name="vector_search",
                success=False,
                result=None,
                error=str(e)
            )
    
    def grep_search(self, pattern: str, case_sensitive: bool = False, max_results: int = 10) -> ToolResult:
        """Execute grep pattern search"""
        try:
            results = self.grep_tool.search(pattern, case_sensitive, max_results)
            return ToolResult(
                tool_name="grep_search",
                success=True,
                result=results
            )
        except Exception as e:
            return ToolResult(
                tool_name="grep_search",
                success=False,
                result=None,
                error=str(e)
            )
    
    def exact_search(self, text: str, case_sensitive: bool = False, max_results: int = 10) -> ToolResult:
        """Execute exact text search"""
        try:
            results = self.grep_tool.search_exact(text, case_sensitive, max_results)
            return ToolResult(
                tool_name="exact_search",
                success=True,
                result=results
            )
        except Exception as e:
            return ToolResult(
                tool_name="exact_search",
                success=False,
                result=None,
                error=str(e)
            )
    
    def keyword_search(self, keywords: List[str], case_sensitive: bool = False, 
                      match_all: bool = False, max_results: int = 10) -> ToolResult:
        """Execute keyword search"""
        try:
            results = self.grep_tool.search_keywords(keywords, case_sensitive, match_all, max_results)
            return ToolResult(
                tool_name="keyword_search",
                success=True,
                result=results
            )
        except Exception as e:
            return ToolResult(
                tool_name="keyword_search",
                success=False,
                result=None,
                error=str(e)
            )
    
    def _format_search_results(self, results: List[ToolResult]) -> str:
        """Format search results for LLM context"""
        formatted = []
        
        for result in results:
            if not result.success:
                formatted.append(f"Tool {result.tool_name} failed: {result.error}")
                continue
            
            if result.tool_name == "vector_search":
                formatted.append("=== Vector Search Results ===")
                for i, item in enumerate(result.result[:3]):  # Top 3 results
                    formatted.append(f"Result {i+1} (score: {item['score']:.3f}):")
                    formatted.append(f"{item['document'][:500]}...")
                    formatted.append("")
            
            elif result.tool_name in ["grep_search", "exact_search"]:
                formatted.append(f"=== {result.tool_name.title()} Results ===")
                for i, item in enumerate(result.result[:3]):  # Top 3 results
                    formatted.append(f"Match {i+1}:")
                    if 'matches' in item:
                        for match in item['matches'][:2]:
                            formatted.append(f"  Context: {match['context']}")
                    elif 'occurrences' in item:
                        for occ in item['occurrences'][:2]:
                            formatted.append(f"  Context: {occ['context']}")
                    formatted.append("")
            
            elif result.tool_name == "keyword_search":
                formatted.append("=== Keyword Search Results ===")
                for i, item in enumerate(result.result[:3]):  # Top 3 results
                    formatted.append(f"Match {i+1} (keywords: {item['found_keywords']}):")
                    formatted.append(f"{item['document'][:500]}...")
                    formatted.append("")
        
        return "\n".join(formatted)
    
    def answer_question(self, question: str, use_vector: bool = True, use_grep: bool = True, 
                       grep_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Answer a question using RAG with available tools"""
        start_time = time.time()
        
        # Collect search results
        search_results = []
        
        if use_vector:
            vector_result = self.vector_search(question, top_k=5)
            search_results.append(vector_result)
        
        if use_grep and grep_patterns:
            for pattern in grep_patterns:
                grep_result = self.grep_search(pattern, case_sensitive=False, max_results=5)
                search_results.append(grep_result)
        
        # Format context
        context = self._format_search_results(search_results)
        
        # Prepare prompt
        system_prompt = """You are a helpful AI assistant with access to a document collection. 
You have been provided with relevant context from the documents to answer the user's question.
Use the provided context to give accurate, detailed answers. If the context doesn't contain 
enough information to answer the question, say so clearly."""

        user_prompt = f"""Question: {question}

Relevant Context:
{context}

Please provide a comprehensive answer based on the available context."""

        # Generate response
        try:
            response = self.client.chat.completions.create(
                model=self.active_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2048
            )
            
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else 0
            
        except Exception as e:
            answer = f"Error generating response: {str(e)}"
            tokens_used = 0
        
        duration = time.time() - start_time
        
        return {
            "question": question,
            "answer": answer,
            "model": self.active_model,
            "context": context,
            "search_results": [result.dict() for result in search_results],
            "tokens_used": tokens_used,
            "duration": duration
        }
    
    def save_index(self, filepath: str):
        """Save vector index"""
        self.vector_tool.save_index(filepath)
    
    def load_index(self, filepath: str):
        """Load vector index"""
        self.vector_tool.load_index(filepath)