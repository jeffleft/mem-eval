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
    """Simple RAG agent with dynamic tool usage"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "o3"):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.vector_tool = VectorSearchTool(api_key=api_key)
        self.grep_tool = GrepTool()
        
        # Define tools for function calling
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "vector_search",
                    "description": "Search for semantically similar content using vector embeddings. Use this for conceptual or topic-based searches.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find semantically similar content"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of top results to return (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "grep_search",
                    "description": "Search for exact patterns or regular expressions in the text. Use this for finding specific phrases, names, dates, or regex patterns.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "The regex pattern or exact text to search for"
                            },
                            "case_sensitive": {
                                "type": "boolean",
                                "description": "Whether the search should be case sensitive (default: false)",
                                "default": False
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return (default: 10)",
                                "default": 10
                            }
                        },
                        "required": ["pattern"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "exact_search", 
                    "description": "Search for exact text matches. Use this when you need to find specific phrases or quotes exactly as written.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The exact text to search for"
                            },
                            "case_sensitive": {
                                "type": "boolean", 
                                "description": "Whether the search should be case sensitive (default: false)",
                                "default": False
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return (default: 10)",
                                "default": 10
                            }
                        },
                        "required": ["text"]
                    }
                }
            }
        ]
    
    def load_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """Load documents into both search tools"""
        self.vector_tool.build_index(documents, metadata)
        self.grep_tool.load_documents(documents, metadata)
    
    def _execute_tool_call(self, tool_call) -> ToolResult:
        """Execute a tool call and return the result"""
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        try:
            if function_name == "vector_search":
                query = function_args["query"]
                top_k = function_args.get("top_k", 5)
                results = self.vector_tool.search(query, top_k)
                return ToolResult(
                    tool_name="vector_search",
                    success=True,
                    result=results
                )
            
            elif function_name == "grep_search":
                pattern = function_args["pattern"]
                case_sensitive = function_args.get("case_sensitive", False)
                max_results = function_args.get("max_results", 10)
                results = self.grep_tool.search(pattern, case_sensitive, max_results)
                return ToolResult(
                    tool_name="grep_search", 
                    success=True,
                    result=results
                )
            
            elif function_name == "exact_search":
                text = function_args["text"]
                case_sensitive = function_args.get("case_sensitive", False)
                max_results = function_args.get("max_results", 10)
                results = self.grep_tool.search_exact(text, case_sensitive, max_results)
                return ToolResult(
                    tool_name="exact_search",
                    success=True,
                    result=results
                )
            
            else:
                return ToolResult(
                    tool_name=function_name,
                    success=False,
                    result=None,
                    error=f"Unknown function: {function_name}"
                )
                
        except Exception as e:
            return ToolResult(
                tool_name=function_name,
                success=False,
                result=None,
                error=str(e)
            )
    
    def _format_tool_results(self, tool_results: List[ToolResult]) -> str:
        """Format tool results for inclusion in context"""
        if not tool_results:
            return "No search results available."
        
        formatted = []
        for result in tool_results:
            if not result.success:
                formatted.append(f"❌ {result.tool_name} failed: {result.error}")
                continue
                
            if result.tool_name == "vector_search":
                formatted.append("=== Vector Search Results ===")
                for i, item in enumerate(result.result[:3]):
                    formatted.append(f"Result {i+1} (similarity: {item['score']:.3f}):")
                    formatted.append(f"{item['document'][:400]}...")
                    formatted.append("")
            
            elif result.tool_name in ["grep_search", "exact_search"]:
                formatted.append(f"=== {result.tool_name.replace('_', ' ').title()} Results ===")
                for i, item in enumerate(result.result[:3]):
                    formatted.append(f"Match {i+1}:")
                    if 'matches' in item:
                        for match in item['matches'][:2]:
                            formatted.append(f"  Context: {match['context']}")
                    elif 'occurrences' in item:
                        for occ in item['occurrences'][:2]:
                            formatted.append(f"  Context: {occ['context']}")
                    formatted.append("")
        
        return "\n".join(formatted)
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using RAG with dynamic tool selection"""
        start_time = time.time()
        
        # First, let the LLM decide which tools to use
        system_prompt = """You are a helpful AI assistant with access to a document collection through search tools.

Available tools:
1. vector_search: For finding conceptually similar content using semantic search
2. grep_search: For finding exact patterns, regex matches, specific phrases, names, dates
3. exact_search: For finding exact text matches

When given a question, first analyze what information you need and use the appropriate tools to gather relevant context. Then provide a comprehensive answer based on the retrieved information."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please help me answer this question: {question}"}
        ]
        
        tool_results = []
        max_iterations = 7  # Prevent infinite loops
        iteration = 0
        final_answer = "No answer provided"
        tokens_used = 0
        
        # Mock the first vector search using real search
        if iteration == 0:
            # Use real vector search with the question
            real_tool_result = self.vector_tool.search(question, 5)
            mock_tool_result = ToolResult(
                tool_name="vector_search",
                success=True,
                result=real_tool_result
            )
            tool_results.append(mock_tool_result)
            
            # Add mock tool call and result to messages
            mock_tool_call = {
                'id': 'gather_context',
                'type': 'function',
                'function': {
                    'name': 'vector_search',
                    'arguments': '{"query":"' + question + '","top_k":5}'
                }
            }
            
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [mock_tool_call]
            })
            messages.append({
                "tool_call_id": "gather_context",
                "role": "tool",
                "name": "vector_search",
                "content": json.dumps(mock_tool_result.dict())
            })
        
        while iteration < max_iterations:
            # Update system prompt with current iteration info
            if iteration > 0:
                current_system_prompt = system_prompt + "\n\nNote that you are on " + str(iteration + 1) + " of a maximum of " + str(max_iterations) + " iterations."
            else:
                current_system_prompt = system_prompt
            
            # Add final iteration instruction
            if iteration == max_iterations - 1:
                current_system_prompt += "\n\nThis is your final iteration (no more tools can be used). Provide a comprehensive answer to the best of your ability based on all the information you've gathered."
            
                # Update the system message
                messages[0] = {"role": "system", "content": current_system_prompt}
                
                # Call the model
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
            else:
               # Update the system message
                messages[0] = {"role": "system", "content": current_system_prompt}
                
                # Call the model with function calling enabled
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                )
            
            response_message = response.choices[0].message
            
            # Check if the model wants to call tools
            if response_message.tool_calls:
                # Execute each tool call
                for tool_call in response_message.tool_calls:
                    tool_result = self._execute_tool_call(tool_call)
                    tool_results.append(tool_result)
                    
                    # Add tool result to conversation
                    messages.append(response_message)
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": json.dumps(tool_result.dict())
                    })
                
                iteration += 1
                continue  # Continue the conversation
            
            else:
                # Model provided final answer
                final_answer = response_message.content
                tokens_used = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else 0
                break
        
        duration = time.time() - start_time
        context = self._format_tool_results(tool_results)
        
        return {
            "question": question,
            "answer": final_answer,
            "model": self.model,
            "context": context,
            "tool_results": [result.dict() for result in tool_results],
            "tokens_used": tokens_used,
            "duration": duration,
            "iterations": iteration,
            "full_dialog": messages
        }
    
    def save_index(self, filepath: str):
        """Save vector index"""
        self.vector_tool.save_index(filepath)
    
    def load_index(self, filepath: str):
        """Load vector index"""
        self.vector_tool.load_index(filepath)