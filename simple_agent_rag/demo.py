#!/usr/bin/env python3
"""
Demo script for Simple Agent RAG
Shows how to use the system with real OpenAI API
"""

import os
import sys
import time
from datetime import datetime
import json

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, try to load .env manually
    def load_dotenv():
        try:
            with open('.env', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
        except FileNotFoundError:
            pass
    
    load_dotenv()

def demo_with_real_api():
    """Demo using real OpenAI API"""
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No OpenAI API key found!")
        print("Please set your API key in one of these ways:")
        print("1. Environment variable:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("2. Or create a .env file in the project root:")
        print("   OPENAI_API_KEY=your-api-key-here")
        return False
    
    try:
        from rag_agent import RAGAgent
        
        model = "o3"
        print(f"🚀 Initializing RAG agent with {model}...")
        agent = RAGAgent(api_key=api_key, model=model)
        print(f"✅ Agent initialized with model: {agent.model}")
        
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
        print("🤖 TESTING DYNAMIC RAG AGENT")
        print("💡 The LLM will dynamically choose which tools to use for each question")
        print("="*60)
        
        for i, question in enumerate(questions, 1):
            print(f"\n💭 Question {i}: {question}")
            print("-" * 50)
            
            start_time = time.time()
            
            # Use new dynamic tool calling approach
            response = agent.answer_question(question)
            
            duration = time.time() - start_time
            
            print(f"📝 Answer: {response['answer']}")
            print(f"⚡ Model: {response['model']}")
            print(f"⏱️  Time: {duration:.2f}s")
            print(f"🔄 Iterations: {response['iterations']}")
            print(f"🛠️  Tools used: {len(response['tool_results'])}")
            
            # Show which tools were dynamically selected
            if response['tool_results']:
                tool_names = [tr['tool_name'] for tr in response['tool_results']]
                print(f"🔍 LLM chose: {', '.join(tool_names)}")
            
            # Show cost info if available
            if response.get('tokens_used'):
                print(f"🪙 Tokens used: {response['tokens_used']}")

            # dump the full dialog to a file
            with open(f"dialog_{i}.json", "w") as f:
                # Convert messages to serializable format
                serializable_messages = []
                for msg in response['full_dialog']:
                    if isinstance(msg, dict):
                        # Already a dictionary (system, user messages)
                        serializable_messages.append(msg)
                    else:
                        # ChatCompletionMessage object - convert to dict
                        msg_dict = {
                            'role': msg.role,
                            'content': msg.content
                        }
                        
                        # Add tool_calls if present
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            msg_dict['tool_calls'] = []
                            for tool_call in msg.tool_calls:
                                tool_call_dict = {
                                    'id': tool_call.id,
                                    'type': tool_call.type,
                                    'function': {
                                        'name': tool_call.function.name,
                                        'arguments': tool_call.function.arguments
                                    }
                                }
                                msg_dict['tool_calls'].append(tool_call_dict)
                        
                        # Add tool_call_id and name if present
                        if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                            msg_dict['tool_call_id'] = msg.tool_call_id
                            msg_dict['name'] = getattr(msg, 'name', None)
                        
                        serializable_messages.append(msg_dict)
                
                json.dump(serializable_messages, f, indent=2)

        print("\n" + "="*60)
        print("✅ DYNAMIC RAG DEMO COMPLETED!")
        print("="*60)
        print("\n🎯 What happened:")
        print("1. 📊 Documents were embedded and indexed using OpenAI embeddings + FAISS")
        print("2. 🧠 The LLM INTELLIGENTLY CHOSE which tools to use for each question")
        print("3. 🔍 Tools were called with LLM-generated parameters (no predefined patterns!)")
        print("4. 🤖 Multi-turn conversations allowed iterative tool usage")
        print("5. ⚡ Real-time performance and cost metrics were tracked")
        
        print("\n🚀 Key Improvement:")
        print("🧠 LLM now acts as an autonomous agent that dynamically selects tools!")
        print("🔍 No more predefined grep patterns - the LLM chooses what to search for!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demo function"""
    print("🎯 Simple Agent RAG Demo")
    print("=" * 40)
    
    # Check if API key is available
    if os.getenv("OPENAI_API_KEY"):
        print("🔑 OpenAI API key found - running live demo...")
        demo_with_real_api()
    else:
        print("ℹ️  No OpenAI API key found - showing instructions...")
        print()
        print("To run this demo with real OpenAI API:")
        print("1. Get your API key from https://platform.openai.com/api-keys")
        print("2. Set it in one of these ways:")
        print("   a) Environment variable:")
        print("      export OPENAI_API_KEY='your-api-key-here'")
        print("   b) Or create a .env file in the project root:")
        print("      OPENAI_API_KEY=your-api-key-here")
        print("3. Run this demo again:")
        print("   python3 demo.py")
        print()
        print("For testing without API key, run:")
        print("   python3 test_rag_agent_mock.py")
        print()

if __name__ == "__main__":
    main()