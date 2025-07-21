#!/usr/bin/env python3
"""
Demo script for Simple Agent RAG with o3
Shows how to use the system with real OpenAI API
"""

import os
import sys
import time
from datetime import datetime

def demo_with_real_api():
    """Demo using real OpenAI API"""
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No OpenAI API key found!")
        print("Please set your API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    try:
        from rag_agent import RAGAgent
        
        print("🚀 Initializing RAG agent with o3...")
        agent = RAGAgent(api_key=api_key, model="o3")
        print(f"✅ Agent initialized with model: {agent.active_model}")
        
        # Sample long conversation documents
        documents = [
            "[2024-01-15 10:00] user: Hi, I'm planning a trip to Paris next month. Any recommendations?",
            "[2024-01-15 10:01] assistant: Paris is wonderful! I recommend visiting the Eiffel Tower, Louvre Museum, and trying some authentic croissants.",
            "[2024-01-15 10:02] user: Great! I love French pastries. What about accommodation?",
            "[2024-01-15 10:03] assistant: For accommodation, the Marais district is charming with many boutique hotels. The Latin Quarter is also great for a cultural experience.",
            "[2024-01-16 14:30] user: I booked a hotel in Marais! When is the best time to visit Eiffel Tower?",
            "[2024-01-16 14:31] assistant: Early morning (7-9 AM) or evening (7-9 PM) are best to avoid crowds. The tower is beautifully lit at night!",
            "[2024-01-16 14:32] user: Perfect! I also want to visit some museums. Which ones should I prioritize?",
            "[2024-01-16 14:33] assistant: The Louvre is a must-see, but book in advance. Musée d'Orsay has incredible Impressionist art. For modern art, try Centre Pompidou.",
            "[2024-01-17 09:15] user: I'm also interested in French cuisine. Any restaurant recommendations?",
            "[2024-01-17 09:16] assistant: For fine dining, try L'Ambroisie or Guy Savoy. For casual but excellent food, visit L'As du Fallafel in Marais or Pierre Hermé for macarons.",
            "[2024-01-17 09:17] user: What about day trips from Paris?",
            "[2024-01-17 09:18] assistant: Versailles is the most popular day trip - take the RER C train. Giverny (Monet's gardens) is beautiful in spring/summer. Fontainebleau castle is also lovely.",
            "[2024-01-18 16:45] user: I'm excited about Versailles! How long should I spend there?",
            "[2024-01-18 16:46] assistant: Plan for a full day at Versailles. The palace tour takes 2-3 hours, and the gardens are vast. Bring comfortable shoes and consider renting a bike for the gardens.",
            "[2024-01-18 16:47] user: Thanks! One last question - what's the weather like in February?",
            "[2024-01-18 16:48] assistant: February in Paris is cool (5-8°C) and often rainy. Pack layers, a waterproof jacket, and comfortable walking shoes. Museums are perfect for rainy days!",
        ]
        
        print(f"📚 Loading {len(documents)} conversation documents...")
        agent.load_documents(documents)
        print("✅ Documents loaded and indexed")
        
        # Test questions
        questions = [
            "What time should I visit the Eiffel Tower?",
            "Which museums did we discuss for my Paris trip?",
            "What restaurant recommendations were given for fine dining?",
            "How should I prepare for the weather in February in Paris?",
            "What day trip options are available from Paris?"
        ]
        
        print("\n" + "="*60)
        print("🤖 TESTING RAG AGENT WITH SAMPLE QUESTIONS")
        print("="*60)
        
        for i, question in enumerate(questions, 1):
            print(f"\n💭 Question {i}: {question}")
            print("-" * 50)
            
            start_time = time.time()
            
            # Create grep patterns
            grep_patterns = []
            if "eiffel" in question.lower():
                grep_patterns = ["Eiffel", "tower", "time", "visit"]
            elif "museum" in question.lower():
                grep_patterns = ["museum", "Louvre", "art"]
            elif "restaurant" in question.lower():
                grep_patterns = ["restaurant", "dining", "food"]
            elif "weather" in question.lower():
                grep_patterns = ["weather", "February", "rain"]
            elif "day trip" in question.lower():
                grep_patterns = ["Versailles", "trip", "train"]
            
            response = agent.answer_question(
                question=question,
                use_vector=True,
                use_grep=True,
                grep_patterns=grep_patterns
            )
            
            duration = time.time() - start_time
            
            print(f"📝 Answer: {response['answer']}")
            print(f"⚡ Model: {response['model']}")
            print(f"⏱️  Time: {duration:.2f}s")
            print(f"🔍 Sources found: {len(response['search_results'])} tools used")
            
        print("\n" + "="*60)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_benchmark_instructions():
    """Show instructions for running the full benchmark"""
    print("\n" + "="*60)
    print("📊 RUNNING THE LONGMEMEVAL BENCHMARK")
    print("="*60)
    print()
    print("To run the full benchmark against LongMemEval:")
    print()
    print("1. Set your OpenAI API key:")
    print("   export OPENAI_API_KEY='your-api-key-here'")
    print()
    print("2. Run quick test:")
    print("   python3 run_benchmark.py --test-only")
    print()
    print("3. Run small benchmark (50 samples):")
    print("   python3 run_benchmark.py --num-samples 50")
    print()
    print("4. Run full benchmark (all samples):")
    print("   python3 run_benchmark.py --num-samples 1000")
    print()
    print("5. Use specific model:")
    print("   python3 run_benchmark.py --model gpt-4o --num-samples 100")
    print()
    print("Available models (in order of preference):")
    print("  - o3 (latest reasoning model)")
    print("  - o1")
    print("  - o1-preview")
    print("  - gpt-4o")
    print("  - gpt-4o-mini")
    print()
    print("The system will automatically fall back to available models.")
    print("="*60)

def main():
    """Main demo function"""
    print("🎯 Simple Agent RAG with o3 Demo")
    print("=" * 40)
    
    # Check if API key is available
    if os.getenv("OPENAI_API_KEY"):
        print("🔑 OpenAI API key found - running live demo...")
        success = demo_with_real_api()
        if success:
            show_benchmark_instructions()
    else:
        print("ℹ️  No OpenAI API key found - showing instructions...")
        print()
        print("To run this demo with real OpenAI API:")
        print("1. Get your API key from https://platform.openai.com/api-keys")
        print("2. Set it as an environment variable:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("3. Run this demo again:")
        print("   python3 demo.py")
        print()
        print("For testing without API key, run:")
        print("   python3 test_rag_agent_mock.py")
        print()
        show_benchmark_instructions()

if __name__ == "__main__":
    main()