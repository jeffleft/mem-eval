#!/usr/bin/env python3
"""
Cost estimation for running the RAG agent benchmark
"""

import json
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ModelPricing:
    """Pricing information for OpenAI models"""
    input_price_per_1k: float  # USD per 1k input tokens
    output_price_per_1k: float  # USD per 1k output tokens
    context_window: int  # Maximum context window

# OpenAI pricing as of December 2024 (you should verify current pricing)
MODEL_PRICING = {
    "o3": ModelPricing(input_price_per_1k=0.060, output_price_per_1k=0.240, context_window=200000),
    "o1": ModelPricing(input_price_per_1k=0.015, output_price_per_1k=0.060, context_window=200000),
    "o1-preview": ModelPricing(input_price_per_1k=0.015, output_price_per_1k=0.060, context_window=128000),
    "gpt-4o": ModelPricing(input_price_per_1k=0.0025, output_price_per_1k=0.010, context_window=128000),
    "gpt-4o-mini": ModelPricing(input_price_per_1k=0.000150, output_price_per_1k=0.000600, context_window=128000),
    "text-embedding-3-small": ModelPricing(input_price_per_1k=0.00002, output_price_per_1k=0.0, context_window=8191),
}

class CostEstimator:
    """Estimates costs for running the RAG benchmark"""
    
    def __init__(self):
        self.pricing = MODEL_PRICING
    
    def estimate_single_query_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a single query"""
        if model not in self.pricing:
            print(f"⚠️ Unknown model {model}, using gpt-4o pricing as fallback")
            model = "gpt-4o"
        
        pricing = self.pricing[model]
        input_cost = (input_tokens / 1000) * pricing.input_price_per_1k
        output_cost = (output_tokens / 1000) * pricing.output_price_per_1k
        return input_cost + output_cost
    
    def estimate_embedding_cost(self, num_documents: int, avg_doc_length: int = 500) -> float:
        """Estimate cost for embedding documents"""
        # Estimate tokens (roughly 4 chars per token)
        total_tokens = (num_documents * avg_doc_length) // 4
        pricing = self.pricing["text-embedding-3-small"]
        return (total_tokens / 1000) * pricing.input_price_per_1k
    
    def estimate_benchmark_cost(self, num_samples: int, model: str = "o3", 
                               avg_input_tokens: int = 3000, avg_output_tokens: int = 500,
                               num_documents: int = 1000, avg_doc_length: int = 500) -> Dict[str, float]:
        """Estimate total cost for running the benchmark"""
        
        # Cost for embeddings (one-time)
        embedding_cost = self.estimate_embedding_cost(num_documents, avg_doc_length)
        
        # Cost for main model queries
        query_cost = self.estimate_single_query_cost(model, avg_input_tokens, avg_output_tokens)
        total_query_cost = query_cost * num_samples
        
        # Cost for grading (assume gpt-4o-mini for grading)
        grading_cost = self.estimate_single_query_cost("gpt-4o-mini", 1000, 50)  # Smaller grading queries
        total_grading_cost = grading_cost * num_samples
        
        total_cost = embedding_cost + total_query_cost + total_grading_cost
        
        return {
            "embedding_cost": embedding_cost,
            "query_cost_per_sample": query_cost,
            "total_query_cost": total_query_cost,
            "grading_cost_per_sample": grading_cost,
            "total_grading_cost": total_grading_cost,
            "total_cost": total_cost,
            "model": model,
            "num_samples": num_samples
        }
    
    def print_cost_breakdown(self, estimate: Dict[str, float]):
        """Print a detailed cost breakdown"""
        print("\n" + "="*50)
        print("💰 COST ESTIMATION")
        print("="*50)
        print(f"Model: {estimate['model']}")
        print(f"Number of samples: {estimate['num_samples']}")
        print()
        print("Cost Breakdown:")
        print(f"  📊 Embeddings (one-time):     ${estimate['embedding_cost']:.4f}")
        print(f"  🤖 Queries ({estimate['num_samples']} × ${estimate['query_cost_per_sample']:.4f}): ${estimate['total_query_cost']:.4f}")
        print(f"  ✅ Grading ({estimate['num_samples']} × ${estimate['grading_cost_per_sample']:.4f}): ${estimate['total_grading_cost']:.4f}")
        print(f"  {'─'*40}")
        print(f"  💵 TOTAL ESTIMATED COST:      ${estimate['total_cost']:.4f}")
        print()
        
        # Cost per sample breakdown
        cost_per_sample = estimate['total_cost'] / estimate['num_samples']
        print(f"📈 Cost per sample: ${cost_per_sample:.4f}")
        
        # Scaling estimates
        print("\n📊 Scaling Estimates:")
        for scale in [10, 50, 100, 500, 1000]:
            if scale != estimate['num_samples']:
                scaled_cost = self.estimate_benchmark_cost(
                    scale, estimate['model']
                )['total_cost']
                print(f"  {scale:4d} samples: ${scaled_cost:.2f}")

def main():
    """Example cost estimation"""
    estimator = CostEstimator()
    
    print("🧮 RAG Benchmark Cost Estimation")
    
    # Different scenarios
    scenarios = [
        {"name": "Small Test (o3)", "samples": 10, "model": "o3"},
        {"name": "Medium Test (o3)", "samples": 50, "model": "o3"},
        {"name": "Full Test (o3)", "samples": 100, "model": "o3"},
        {"name": "Budget Test (gpt-4o-mini)", "samples": 100, "model": "gpt-4o-mini"},
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*20} {scenario['name']} {'='*20}")
        estimate = estimator.estimate_benchmark_cost(
            num_samples=scenario["samples"],
            model=scenario["model"]
        )
        estimator.print_cost_breakdown(estimate)

if __name__ == "__main__":
    main()