#!/usr/bin/env python3
"""
Test script demonstrating dynamic tool usage by the RAG agent
"""

import os
import sys
from unittest.mock import Mock, patch
import numpy as np
import json

# Mock OpenAI for testing without API key
class MockToolCall:
    def __init__(self, id, function_name, arguments):
        self.id = id
        self.function = Mock()
        self.function.name = function_name
        self.function.arguments = json.dumps(arguments)

class MockMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

class MockChoice:
    def __init__(self, message):
        self.message = message

class MockUsage:
    def __init__(self, total_tokens=100):
        self.total_tokens = total_tokens

class MockResponse:
    def __init__(self, content=None, tool_calls=None, usage=None):
        self.choices = [MockChoice(MockMessage(content, tool_calls))]
        self.usage = usage or MockUsage()

class MockOpenAI:
    def __init__(self, *args, **kwargs):
        self.call_count = 0
    
    @property
    def chat(self):
        return self
    
    @property
    def completions(self):
        return self
    
    @property
    def embeddings(self):
        return self
    
    def create(self, *args, **kwargs):
        self.call_count += 1
        
        # Handle embeddings
        if 'input' in kwargs:
            return Mock(data=[Mock(embedding=np.random.random(1536).tolist())])
        
        # Handle function calling scenario
        if 'tools' in kwargs and self.call_count == 1:
            # First call - model decides to use tools
            tool_calls = [
                MockToolCall("call_1", "vector_search", {"query": "John Paris trip", "top_k": 5}),
                MockToolCall("call_2", "grep_search", {"pattern": "croissant", "case_sensitive": False})
            ]
            return MockResponse(content=None, tool_calls=tool_calls)
        
        elif 'tools' in kwargs and self.call_count == 2:
            # Second call - model provides final answer
            return MockResponse(
                content="Based on the search results, John visited Paris on 2024-01-15 and enjoyed croissants from a local bakery.",
                tool_calls=None
            )
        
        # Fallback for other calls
        return MockResponse(content="Test response")

def test_dynamic_rag():
    """Test the dynamic RAG agent with mock OpenAI"""
    
    with patch('rag_agent.OpenAI', MockOpenAI), \
         patch('vector_search_tool.OpenAI', MockOpenAI):
        from rag_agent import RAGAgent
        
        print("🧪 Testing Dynamic RAG Agent")
        print("="*50)
        
        # Initialize agent
        agent = RAGAgent()
        print(f"✅ Agent initialized with model: {agent.active_model}")
        
        # Load test documents
        documents = [
            "John visited Paris on 2024-01-15. He enjoyed the Eiffel Tower and ate croissants.",
            "Mary went to London on 2024-02-10. She saw Big Ben and visited the British Museum.",
            "The weather in Paris was sunny with a temperature of 20°C on January 15th, 2024.",
            "John's favorite food during his Paris trip was croissants from a local bakery.",
            "Mary bought souvenirs in London including a red telephone booth miniature."
        ]
        
        agent.load_documents(documents)
        print(f"✅ Loaded {len(documents)} documents")
        
        # Test dynamic question answering
        test_questions = [
            "What did John eat in Paris?",
            "When did Mary visit London?",
            "What was the weather like during John's trip?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 Test {i}: {question}")
            print("-" * 30)
            
            # Mock the agent's client for this specific test
            agent.client = MockOpenAI()
            
            try:
                result = agent.answer_question(question)
                
                print(f"🤖 Model: {result['model']}")
                print(f"⏱️  Duration: {result['duration']:.2f}s")
                print(f"🔄 Iterations: {result['iterations']}")
                print(f"🎯 Answer: {result['answer']}")
                print(f"🛠️  Tools used: {len(result['tool_results'])}")
                
                # Show which tools were called
                for tool_result in result['tool_results']:
                    tool_name = tool_result['tool_name']
                    success = tool_result['success']
                    status = "✅" if success else "❌"
                    print(f"   {status} {tool_name}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n" + "="*50)
        print("✅ Dynamic RAG test completed!")
        
        # Show the key improvements
        print("\n🚀 Key Improvements:")
        print("1. 🧠 LLM now dynamically chooses which tools to use")
        print("2. 🔍 LLM generates its own search patterns and keywords")
        print("3. 🎯 Tools are called based on question analysis, not pre-defined patterns")
        print("4. 🔄 Multi-turn conversation allows for iterative tool usage")
        print("5. 💡 Function calling provides structured tool interaction")

def show_tool_descriptions():
    """Show the tool descriptions that the LLM sees"""
    print("\n🛠️  Available Tools for LLM:")
    print("="*50)
    
    tools = [
        {
            "name": "vector_search",
            "description": "Search for semantically similar content using vector embeddings. Use this for conceptual or topic-based searches.",
            "example": "vector_search(query='John Paris trip', top_k=5)"
        },
        {
            "name": "grep_search", 
            "description": "Search for exact patterns or regular expressions in the text. Use this for finding specific phrases, names, dates, or regex patterns.",
            "example": "grep_search(pattern='croissant', case_sensitive=False)"
        },
        {
            "name": "exact_search",
            "description": "Search for exact text matches. Use this when you need to find specific phrases or quotes exactly as written.",
            "example": "exact_search(text='2024-01-15', case_sensitive=False)"
        }
    ]
    
    for tool in tools:
        print(f"📍 {tool['name']}")
        print(f"   Description: {tool['description']}")
        print(f"   Example: {tool['example']}")
        print()

if __name__ == "__main__":
    test_dynamic_rag()
    show_tool_descriptions()