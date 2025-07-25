#!/usr/bin/env python3
"""
Utility script to analyze message history from benchmark runs
"""

import json
import argparse
from typing import Dict, List, Any
from collections import Counter

def load_benchmark_results(filename: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def analyze_message_history(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze message history patterns"""
    detailed_results = results.get('detailed_results', [])
    
    if not detailed_results:
        return {"error": "No detailed results found"}
    
    # Basic statistics
    total_samples = len(detailed_results)
    samples_with_history = sum(1 for r in detailed_results if r.get('message_history'))
    
    # Message statistics
    message_counts = [len(r.get('message_history', [])) for r in detailed_results]
    tool_call_counts = [len(r.get('tool_results', [])) for r in detailed_results]
    iteration_counts = [r.get('iterations', 0) for r in detailed_results]
    
    # Tool usage analysis
    tool_usage = Counter()
    for result in detailed_results:
        for tool_result in result.get('tool_results', []):
            tool_usage[tool_result.get('tool_name', 'unknown')] += 1
    
    # Question type analysis
    question_types = Counter()
    for result in detailed_results:
        question_types[result.get('question_type', 'unknown')] += 1
    
    # Grade analysis
    grades = Counter()
    for result in detailed_results:
        grades[str(result.get('grade', False))] += 1
    
    return {
        "total_samples": total_samples,
        "samples_with_history": samples_with_history,
        "message_statistics": {
            "total_messages": sum(message_counts),
            "avg_messages_per_sample": sum(message_counts) / total_samples if total_samples > 0 else 0,
            "min_messages": min(message_counts) if message_counts else 0,
            "max_messages": max(message_counts) if message_counts else 0
        },
        "tool_statistics": {
            "total_tool_calls": sum(tool_call_counts),
            "avg_tool_calls_per_sample": sum(tool_call_counts) / total_samples if total_samples > 0 else 0,
            "tool_usage_breakdown": dict(tool_usage)
        },
        "iteration_statistics": {
            "total_iterations": sum(iteration_counts),
            "avg_iterations_per_sample": sum(iteration_counts) / total_samples if total_samples > 0 else 0,
            "min_iterations": min(iteration_counts) if iteration_counts else 0,
            "max_iterations": max(iteration_counts) if iteration_counts else 0
        },
        "question_type_distribution": dict(question_types),
        "grade_distribution": dict(grades)
    }

def print_analysis(analysis: Dict[str, Any]):
    """Print formatted analysis results"""
    print("=" * 60)
    print("MESSAGE HISTORY ANALYSIS")
    print("=" * 60)
    
    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    print(f"Total Samples: {analysis['total_samples']}")
    print(f"Samples with Message History: {analysis['samples_with_history']}")
    
    print("\n" + "-" * 40)
    print("MESSAGE STATISTICS")
    print("-" * 40)
    msg_stats = analysis['message_statistics']
    print(f"Total Messages: {msg_stats['total_messages']}")
    print(f"Average Messages per Sample: {msg_stats['avg_messages_per_sample']:.1f}")
    print(f"Min Messages: {msg_stats['min_messages']}")
    print(f"Max Messages: {msg_stats['max_messages']}")
    
    print("\n" + "-" * 40)
    print("TOOL USAGE STATISTICS")
    print("-" * 40)
    tool_stats = analysis['tool_statistics']
    print(f"Total Tool Calls: {tool_stats['total_tool_calls']}")
    print(f"Average Tool Calls per Sample: {tool_stats['avg_tool_calls_per_sample']:.1f}")
    print("\nTool Usage Breakdown:")
    for tool, count in tool_stats['tool_usage_breakdown'].items():
        print(f"  {tool}: {count}")
    
    print("\n" + "-" * 40)
    print("ITERATION STATISTICS")
    print("-" * 40)
    iter_stats = analysis['iteration_statistics']
    print(f"Total Iterations: {iter_stats['total_iterations']}")
    print(f"Average Iterations per Sample: {iter_stats['avg_iterations_per_sample']:.1f}")
    print(f"Min Iterations: {iter_stats['min_iterations']}")
    print(f"Max Iterations: {iter_stats['max_iterations']}")
    
    print("\n" + "-" * 40)
    print("QUESTION TYPE DISTRIBUTION")
    print("-" * 40)
    for qtype, count in analysis['question_type_distribution'].items():
        print(f"  {qtype}: {count}")
    
    print("\n" + "-" * 40)
    print("GRADE DISTRIBUTION")
    print("-" * 40)
    for grade, count in analysis['grade_distribution'].items():
        print(f"  {grade}: {count}")

def main():
    parser = argparse.ArgumentParser(description="Analyze message history from benchmark results")
    parser.add_argument("results_file", help="Path to benchmark results JSON file")
    parser.add_argument("--output", help="Output file for analysis (optional)")
    
    args = parser.parse_args()
    
    try:
        # Load results
        results = load_benchmark_results(args.results_file)
        
        # Analyze message history
        analysis = analyze_message_history(results)
        
        # Print analysis
        print_analysis(analysis)
        
        # Save analysis if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"\nAnalysis saved to: {args.output}")
            
    except FileNotFoundError:
        print(f"Error: File '{args.results_file}' not found")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{args.results_file}'")
    except Exception as e:
        print(f"Error analyzing results: {str(e)}")

if __name__ == "__main__":
    main() 