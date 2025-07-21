import os
from dotenv import load_dotenv
from rag_agent import RAGAgent

def test_rag_agent():
    """Test the RAG agent with sample documents"""
    load_dotenv()
    
    # Sample documents
    documents = [
        "John visited Paris on 2024-01-15. He enjoyed the Eiffel Tower and ate croissants.",
        "Mary went to London on 2024-02-10. She saw Big Ben and visited the British Museum.",
        "The weather in Paris was sunny with a temperature of 20°C on January 15th, 2024.",
        "John's favorite food during his Paris trip was croissants from a local bakery.",
        "Mary bought souvenirs in London including a red telephone booth miniature.",
        "The flight from New York to Paris took 8 hours and cost $800.",
        "Conference schedule: Monday 9 AM - Opening ceremony, Tuesday 2 PM - Panel discussion",
        "Email from boss: Please submit the quarterly report by Friday 5 PM.",
        "Restaurant reservation: Table for 4 at 7:30 PM on Saturday at Le Bernardin",
        "Meeting notes: Discussed budget allocation for Q2, need to reduce costs by 15%"
    ]
    
    metadata = [{"id": i, "source": f"doc_{i}"} for i in range(len(documents))]
    
    print("Initializing RAG agent...")
    agent = RAGAgent()
    
    print("Loading documents...")
    agent.load_documents(documents, metadata)
    
    # Test questions
    questions = [
        "Where did John go and when?",
        "What did Mary buy in London?",
        "What was the weather like in Paris?",
        "Tell me about the conference schedule",
        "What's the deadline for the quarterly report?"
    ]
    
    print("\nTesting RAG agent...")
    print("="*60)
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 40)
        
        # Create simple grep patterns
        grep_patterns = question.lower().split()[:3]  # Simple keyword extraction
        
        try:
            result = agent.answer_question(
                question=question,
                use_vector=True,
                use_grep=True,
                grep_patterns=grep_patterns
            )
            
            print(f"Model: {result['model']}")
            print(f"Answer: {result['answer']}")
            print(f"Duration: {result['duration']:.2f}s")
            print(f"Tokens used: {result['tokens_used']}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*60)
    print("RAG agent test completed!")

if __name__ == "__main__":
    test_rag_agent()