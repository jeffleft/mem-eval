#!/usr/bin/env python3
"""
Main script to run the RAG agent benchmark against LongMemEval
"""

import os
import sys
import argparse
from datetime import datetime
from longmemeval_benchmark import LongMemEvalBenchmark
from test_rag_agent import test_rag_agent

def main():
    parser = argparse.ArgumentParser(description="Run RAG agent benchmark against LongMemEval")
    parser.add_argument("--test-only", action="store_true", help="Run quick test only")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of samples to evaluate (default: 50)")
    parser.add_argument("--model", default="o3", help="Model to use (default: o3)")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Set up API key
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable or use --api-key flag.")
        sys.exit(1)
    
    print(f"Starting RAG agent benchmark at {datetime.now()}")
    print(f"Model: {args.model}")
    print(f"Samples: {args.num_samples}")
    print("-" * 50)
    
    if args.test_only:
        print("Running quick test...")
        test_rag_agent()
        print("Test completed successfully!")
        return
    
    try:
        # Initialize benchmark
        benchmark = LongMemEvalBenchmark(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Download dataset if not present
        print("Downloading LongMemEval dataset...")
        benchmark.download_dataset()
        
        # Load dataset
        print("Loading dataset...")
        benchmark.load_dataset()
        
        # Run benchmark
        print(f"Running benchmark with {args.num_samples} samples...")
        results = benchmark.run_benchmark(
            num_samples=args.num_samples,
            model=args.model
        )
        
        # Print results
        print("\n" + "="*50)
        print("BENCHMARK RESULTS")
        print("="*50)
        print(f"Model: {results['model']}")
        print(f"Total Samples: {results['total_samples']}")
        print(f"Correct Answers: {results['correct_answers']}")
        print(f"Accuracy: {results['accuracy']:.2%}")
        print(f"Average Response Time: {results['avg_response_time']:.2f}s")
        print(f"Total Time: {results['total_time']:.2f}s")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"benchmark_results_{timestamp}.json"
        
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
        
    except Exception as e:
        print(f"Error running benchmark: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()