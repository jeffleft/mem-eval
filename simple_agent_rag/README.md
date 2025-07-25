# Simple Agent RAG with o3 and LongMemEval Benchmark

This implementation provides a simple Retrieval-Augmented Generation (RAG) agent using OpenAI's o3 model (with fallbacks) and benchmarks it against the LongMemEval dataset.

## 🎯 Quick Start

### 1. Testing without API Key
```bash
# Test the implementation with mocked API calls
python3 test_rag_agent_mock.py
```

### 2. Running with Real API Key
```bash
# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'

# Run the demo
python3 demo.py

# Run benchmark test
python3 run_benchmark.py --test-only

# Run small benchmark
python3 run_benchmark.py --num-samples 50
```

## 🏗️ Architecture

### Core Components

1. **RAG Agent** (`rag_agent.py`)
   - Supports OpenAI o3, o1, o1-preview, gpt-4o, and gpt-4o-mini models with automatic fallback
   - Vector similarity search using OpenAI embeddings and FAISS
   - Grep-based pattern search for exact text matching
   - Intelligent model selection based on availability

2. **Vector Search Tool** (`vector_search_tool.py`)
   - Uses OpenAI `text-embedding-3-small` embeddings (1536 dimensions)
   - FAISS IndexFlatIP for efficient cosine similarity search
   - Fallback to sentence-transformers if OpenAI embeddings fail
   - L2 normalization for proper cosine similarity

3. **Grep Tool** (`grep_tool.py`)
   - Regex pattern search
   - Exact text matching
   - Keyword search with AND/OR logic
   - Context extraction around matches

4. **LongMemEval Benchmark** (`longmemeval_benchmark.py`)
   - Automated download and processing of LongMemEval dataset
   - Compatible grading system using the same criteria as the original
   - Comprehensive evaluation metrics
   - Support for different question types

## Installation

1. Clone or download the code
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

### Quick Test

Run a simple test to verify the system works:

```bash
python test_rag_agent.py
```

### Full LongMemEval Benchmark

Run the complete benchmark:

```bash
python longmemeval_benchmark.py
```

### Custom Usage

```python
from rag_agent import RAGAgent

# Initialize agent (will automatically select best available model)
agent = RAGAgent()

# Load your documents
documents = ["Document 1 text...", "Document 2 text..."]
agent.load_documents(documents)

# Ask questions
response = agent.answer_question("Your question here")
print(response['answer'])
print(f"Message history: {len(response['full_dialog'])} messages")
```

## 📊 Message History and Analysis

The benchmark now saves complete message history for each sample, including:
- Full conversation dialog between user and assistant
- Tool calls and their results
- Iteration counts and reasoning steps
- Tool usage patterns

### Analyzing Message History

Use the analysis utility to examine conversation patterns:

```bash
# Analyze a benchmark results file
python analyze_message_history.py benchmark_results_20241201_143022.json

# Save analysis to file
python analyze_message_history.py benchmark_results_20241201_143022.json --output analysis.json
```

The analysis provides:
- Message count statistics
- Tool usage breakdown
- Iteration patterns
- Question type distribution
- Grade distribution
result = agent.answer_question(
    question="What is the main topic?",
    use_vector=True,
    use_grep=True,
    grep_patterns=["topic", "main", "subject"]
)

print(result['answer'])
```

## OpenAI o3 Implementation

### Model Priority

The system attempts to use models in this order:
1. **o3** - Latest reasoning model (preferred)
2. **o1** - Advanced reasoning model
3. **o1-preview** - Preview reasoning model  
4. **gpt-4o** - Multimodal flagship model
5. **gpt-4o-mini** - Efficient flagship model

### o3 Specific Features

Based on the latest OpenAI documentation, o3 offers:
- Enhanced reasoning capabilities for complex problem-solving
- Better performance on mathematical and logical tasks
- Improved instruction following
- Advanced multi-step reasoning

### Fallback Strategy

If o3 is not available (common as it's newly released), the system automatically falls back to the next best available model while maintaining the same API interface.

## Tools Implementation

### Vector Similarity Search

- **Embedding Model**: `text-embedding-3-small` (1536 dimensions)
- **Vector Database**: FAISS with cosine similarity
- **Search Strategy**: Retrieve top-k most similar documents
- **Fallback**: Sentence transformers for offline embedding generation

### Grep Search

- **Pattern Types**: Regex, exact text, keyword search
- **Context Extraction**: 50 characters before/after matches
- **Case Sensitivity**: Configurable
- **Result Limiting**: Prevents overwhelming the LLM with too many matches

## LongMemEval Benchmark

### Dataset

- Automatically downloads from official Google Drive source
- Supports oracle, small (s), and medium (m) datasets
- Processes multi-session conversations into searchable documents

### Evaluation Metrics

- **Accuracy**: Percentage of correct answers
- **Duration**: Total time per question including retrieval
- **Retrieval Duration**: Time spent on document retrieval
- **Question Type Analysis**: Performance breakdown by question category

### Grading

Uses the same grading criteria as the original LongMemEval:
- Temporal reasoning with off-by-one tolerance
- Knowledge update detection
- Single-session preference evaluation
- Default comprehensive answer checking

## Performance Considerations

### Optimization Features

1. **Model Selection**: Automatic fallback ensures availability
2. **Context Management**: Limits retrieved context to prevent token overflow
3. **Caching**: Vector index can be saved/loaded to avoid recomputation
4. **Batching**: Supports processing multiple examples efficiently

### Token Management

- Context length monitoring
- Smart truncation of search results
- Efficient prompt construction
- Model-specific token limits respected

## Results Structure

The benchmark produces comprehensive results including:

```json
{
  "total_examples": 20,
  "correct_answers": 15,
  "accuracy": 0.75,
  "avg_duration": 3.2,
  "avg_retrieval_duration": 1.1,
  "type_statistics": {
    "single-session-assistant": {
      "correct": 15,
      "total": 20,
      "accuracy": 0.75
    }
  },
  "detailed_results": [...] 
}
```

## Customization

### Adding New Tools

Extend the `RAGAgent` class to add new retrieval tools:

```python
def my_custom_search(self, query: str) -> ToolResult:
    # Implement your search logic
    return ToolResult(
        tool_name="custom_search",
        success=True,
        result=results
    )
```

### Model Configuration

Modify the `available_models` list in `RAGAgent.__init__()` to change model priority or add new models.

### Search Strategy

Adjust search parameters in the `answer_question` method:
- Vector search top_k
- Grep search max_results
- Context formatting logic

## Troubleshooting

### Common Issues

1. **API Key**: Ensure OPENAI_API_KEY is set in environment
2. **Model Access**: o3 may require special access - system will fallback automatically
3. **Memory**: Large document sets may require more RAM for FAISS indexing
4. **Rate Limits**: Built-in retry logic handles temporary API limits

### Debug Mode

Add verbose logging to see detailed execution:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Architecture Benefits

### Modular Design
- Separate tools can be used independently
- Easy to swap embedding models or vector databases
- Clear separation between retrieval and generation

### Scalability
- FAISS enables efficient similarity search at scale
- Streaming-compatible design for large datasets
- Memory-efficient document processing

### Robustness
- Multiple fallback strategies
- Error handling throughout the pipeline
- Graceful degradation when components fail

## Future Enhancements

### Potential Improvements
1. **Hybrid Search**: Combine vector and keyword search scores
2. **Query Expansion**: Automatic query reformulation for better retrieval
3. **Caching**: Result caching for repeated queries
4. **Multi-modal**: Support for image and audio documents
5. **Fine-tuning**: Custom embedding models for domain-specific retrieval

This implementation provides a solid foundation for RAG applications while maintaining compatibility with the latest OpenAI models and established evaluation benchmarks.