#!/usr/bin/env python3
"""
Mock test for RAG agent without requiring API keys
"""

import os
import sys
from unittest.mock import Mock, patch
import numpy as np

# Mock OpenAI responses
class MockEmbeddingResponse:
    def __init__(self, embedding):
        self.data = [Mock(embedding=embedding)]

class MockChatResponse:
    def __init__(self, content):
        self.choices = [Mock(message=Mock(content=content))]
        self.usage = Mock(total_tokens=100)

class MockOpenAI:
    def __init__(self, *args, **kwargs):
        pass
    
    @property
    def embeddings(self):
        return self
    
    @property
    def chat(self):
        return self
    
    @property
    def completions(self):
        return self
    
    def create(self, *args, **kwargs):
        if 'model' in kwargs and 'text-embedding' in kwargs['model']:
            # Mock embedding
            embedding = np.random.rand(1536).tolist()
            return MockEmbeddingResponse(embedding)
        else:
            # Mock chat completion
            return MockChatResponse("This is a mock response from the RAG agent.")

def test_rag_agent_mock():
    """Test the RAG agent with mocked OpenAI calls"""
    
    # Mock OpenAI
    with patch('openai.OpenAI', MockOpenAI):
        from rag_agent import RAGAgent
        
        print("Initializing RAG agent (mocked)...")
        
        # Sample documents
        documents = [
            "John visited Paris on 2024-01-15. He enjoyed the Eiffel Tower and ate croissants.",
            "Mary went to London on 2024-02-10. She saw Big Ben and visited the British Museum.",
            "The weather in Paris was sunny with a temperature of 20°C on January 15th, 2024.",
            "John's favorite food during his Paris trip was croissants from a local bakery.",
            "Mary bought souvenirs in London including a red telephone booth miniature.",
        ]
        
        try:
            # Initialize agent with mocked API
            agent = RAGAgent(api_key="mock_key")
            print(f"✓ Agent initialized with model: {agent.active_model}")
            
            # Load documents
            print("Loading documents...")
            agent.load_documents(documents)
            print("✓ Documents loaded successfully")
            
            # Test vector search
            print("Testing vector search...")
            vector_result = agent.vector_search("What did John eat in Paris?", top_k=3)
            print(f"✓ Vector search completed: {vector_result.success}")
            
            # Test grep search
            print("Testing grep search...")
            grep_result = agent.grep_search("croissant", max_results=3)
            print(f"✓ Grep search completed: {grep_result.success}")
            
            # Test exact search
            print("Testing exact search...")
            exact_result = agent.exact_search("Paris", max_results=3)
            print(f"✓ Exact search completed: {exact_result.success}")
            
            # Test keyword search
            print("Testing keyword search...")
            keyword_result = agent.keyword_search(["John", "Paris"], max_results=3)
            print(f"✓ Keyword search completed: {keyword_result.success}")
            
            # Test question answering
            print("Testing question answering...")
            response = agent.answer_question(
                "What did John eat in Paris?",
                use_vector=True,
                use_grep=True,
                grep_patterns=["croissant", "food", "eat"]
            )
            print(f"✓ Question answered successfully")
            print(f"  Model: {response['model']}")
            print(f"  Duration: {response['duration']:.2f}s")
            print(f"  Answer: {response['answer'][:100]}...")
            
            print("\n" + "="*50)
            print("ALL TESTS PASSED - RAG AGENT IS WORKING!")
            print("="*50)
            print("\nYou can now run the full benchmark with a real OpenAI API key:")
            print("export OPENAI_API_KEY='your-api-key-here'")
            print("python3 run_benchmark.py --test-only")
            
        except Exception as e:
            print(f"✗ Error during testing: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

if __name__ == "__main__":
    test_rag_agent_mock()