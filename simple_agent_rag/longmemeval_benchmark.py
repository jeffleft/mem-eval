import asyncio
import json
import gdown
import tarfile
import os
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Any
import time
from openai import OpenAI
from pydantic import BaseModel, Field
from rag_agent import RAGAgent

from dotenv import load_dotenv
load_dotenv()


class Grade(BaseModel):
    is_correct: str = Field(description='yes or no')


class LongMemEvalBenchmark:
    """Benchmark RAG agent against LongMemEval dataset"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.rag_agent = None
        self.dataset_df = None
        
    def download_dataset(self):
        """Download and extract LongMemEval dataset"""
        file_id = '1zJgtYRFhOh5zDQzzatiddfjYhFSnyQ80'
        url = f'https://drive.google.com/uc?id={file_id}'
        file_path = 'longmemeval_data.tar.gz'
        
        # Download the compressed dataset
        if not os.path.exists(file_path):
            print("Downloading LongMemEval dataset...")
            gdown.download(url, file_path, quiet=False)
        else:
            print(f"'{file_path}' already exists, skipping download.")
        
        # Extract the tar.gz
        if not os.path.exists('./longmemeval_oracle.json'):
            print("Extracting dataset...")
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall()
        else:
            print("Dataset already extracted.")
    
    def load_dataset(self, dataset_type: str = 's'):
        """Load LongMemEval dataset"""
        if dataset_type not in ['oracle', 's', 'm']:
            raise ValueError("dataset_type must be 'oracle', 's', or 'm'")
        
        dataset_path = f'data/longmemeval_{dataset_type}.json'
        if not os.path.exists(dataset_path):
            dataset_path = f'longmemeval_{dataset_type}.json'
        
        print(f"Loading dataset from {dataset_path}")
        self.dataset_df = pd.read_json(dataset_path)
        print(f"Loaded {len(self.dataset_df)} examples")
    
    def prepare_documents(self, multi_session_idx: int) -> List[str]:
        """Prepare documents from a multi-session conversation"""
        multi_session = self.dataset_df['haystack_sessions'].iloc[multi_session_idx]
        multi_session_dates = self.dataset_df['haystack_dates'].iloc[multi_session_idx]
        
        documents = []
        for session_idx, session in enumerate(multi_session):
            for msg_idx, msg in enumerate(session):
                date = multi_session_dates[session_idx] + ' UTC'
                date_format = '%Y/%m/%d (%a) %H:%M UTC'
                date_string = datetime.strptime(date, date_format).replace(tzinfo=timezone.utc)
                
                # Create document with metadata
                doc_content = f"[{date_string}] {msg['role']}: {msg['content']}"
                documents.append(doc_content)
        
        return documents
    

    
    def lme_grader(self, question: str, gold_answer: str, response: str, question_type: str) -> bool:
        """Grade the response using LME grading criteria"""
        system_prompt = """You are an expert grader that determines if answers to questions match a gold standard answer"""
        
        TEMPORAL_REASONING_PROMPT = f"""
        I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.

        <QUESTION>
        {question}
        </QUESTION>
        <CORRECT ANSWER>
        {gold_answer}
        </CORRECT ANSWER>
        <RESPONSE>
        {response}
        </RESPONSE>
        """
        
        KNOWLEDGE_UPDATE_PROMPT = f"""
        I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.
        
        <QUESTION>
        {question}
        </QUESTION>
        <CORRECT ANSWER>
        {gold_answer}
        </CORRECT ANSWER>
        <RESPONSE>
        {response}
        </RESPONSE>
        """
        
        SINGLE_SESSION_PREFERENCE = f"""
        I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.
        
        <QUESTION>
        {question}
        </QUESTION>
        <RUBRIC>
        {gold_answer}
        </RUBRIC>
        <RESPONSE>
        {response}
        </RESPONSE>
        """

        DEFAULT_PROMPT = f"""         
        I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.
                
        <QUESTION>
        {question}
        </QUESTION>
        <CORRECT ANSWER>
        {gold_answer}
        </CORRECT ANSWER>
        <RESPONSE>
        {response}
        </RESPONSE>
        """
        
        prompt = DEFAULT_PROMPT
        if question_type == 'temporal-reasoning':
            prompt = TEMPORAL_REASONING_PROMPT
        elif question_type == 'knowledge-update':
            prompt = KNOWLEDGE_UPDATE_PROMPT
        elif question_type == 'single-session-preference':
            prompt = SINGLE_SESSION_PREFERENCE

        try:
            response = self.client.beta.chat.completions.parse(
                model='gpt-4.1-mini',
                messages=[{"role": "system", "content": system_prompt},
                         {"role": "user", "content": prompt}],
                response_format=Grade,
                temperature=0,
            )
            result = response.choices[0].message.parsed
            return result.is_correct.strip().lower() == 'yes'
        except Exception as e:
            print(f"Grading error: {e}")
            return False
    
    def evaluate_single_example(self, multi_session_idx: int) -> Dict[str, Any]:
        """Evaluate a single example from the dataset"""
        start_time = time.time()
        
        # Get question and answer
        question_id = self.dataset_df['question_id'][multi_session_idx]
        question_type = self.dataset_df['question_type'][multi_session_idx]
        question = '(date: ' + self.dataset_df['question_date'][multi_session_idx] + ') ' + self.dataset_df['question'][multi_session_idx]
        gold_answer = self.dataset_df['answer'][multi_session_idx]
        
        # Prepare documents
        documents = self.prepare_documents(multi_session_idx)
        
        # Initialize RAG agent for this example
        self.rag_agent = RAGAgent(api_key=self.api_key)
        self.rag_agent.load_documents(documents)
        
        retrieval_start = time.time()
        
        # Get response from RAG agent (now with dynamic tool selection)
        response_data = self.rag_agent.answer_question(question)
        
        retrieval_duration = time.time() - retrieval_start
        
        # Grade the response
        grade = self.lme_grader(question, gold_answer, response_data['answer'], question_type)
        
        total_duration = time.time() - start_time
        
        result = {
            "question_id": question_id,
            "question_type": question_type,
            "question": question,
            "gold_answer": gold_answer,
            "response": response_data['answer'],
            "model": response_data['model'],
            "grade": grade,
            "context_length": len(response_data['context'].split()),
            "retrieval_duration": retrieval_duration,
            "total_duration": total_duration,
            "tokens_used": response_data.get('tokens_used', 0),
            "message_history": response_data.get('full_dialog', []),
            "tool_results": response_data.get('tool_results', []),
            "iterations": response_data.get('iterations', 0)
        }
        
        return result
    
    def run_benchmark(self, num_samples: int = 50, start_idx: int = 0, 
                     question_types: List[str] = None, model: str = "o3") -> Dict[str, Any]:
        """Run benchmark on specified examples"""
        if self.dataset_df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        # Filter by question types if specified
        if question_types:
            mask = self.dataset_df['question_type'].isin(question_types)
            valid_indices = self.dataset_df[mask].index.tolist()[start_idx:start_idx + num_samples]
        else:
            valid_indices = list(range(start_idx, min(start_idx + num_samples, len(self.dataset_df))))
        
        print(f"Running benchmark on {len(valid_indices)} examples...")
        
        results = []
        correct_count = 0
        
        for i, idx in enumerate(valid_indices):
            print(f"Processing example {i+1}/{len(valid_indices)} (index {idx})")
            
            try:
                result = self.evaluate_single_example(idx)
                results.append(result)
                
                if result['grade']:
                    correct_count += 1
                
                print(f"  Grade: {result['grade']}, Question Type: {result['question_type']}")
                
            except Exception as e:
                print(f"Error processing example {idx}: {e}")
                continue
        
        # Calculate statistics
        accuracy = correct_count / len(results) if results else 0
        avg_duration = sum(r['total_duration'] for r in results) / len(results) if results else 0
        avg_retrieval_duration = sum(r['retrieval_duration'] for r in results) / len(results) if results else 0
        
        # Statistics by question type
        type_stats = {}
        for result in results:
            qtype = result['question_type']
            if qtype not in type_stats:
                type_stats[qtype] = {'correct': 0, 'total': 0}
            type_stats[qtype]['total'] += 1
            if result['grade']:
                type_stats[qtype]['correct'] += 1
        
        for qtype in type_stats:
            type_stats[qtype]['accuracy'] = type_stats[qtype]['correct'] / type_stats[qtype]['total']
        
        benchmark_results = {
            "model": model,
            "total_samples": len(results),
            "correct_answers": correct_count,
            "accuracy": accuracy,
            "avg_response_time": avg_duration,
            "total_time": sum(r['total_duration'] for r in results),
            "avg_retrieval_duration": avg_retrieval_duration,
            "type_statistics": type_stats,
            "detailed_results": results
        }
        
        return benchmark_results
    
    def save_results(self, results: Dict[str, Any], filename: str = "rag_benchmark_results.json"):
        """Save benchmark results to file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {filename}")


def main():
    """Main function to run the benchmark"""
    # Initialize benchmark
    benchmark = LongMemEvalBenchmark()
    
    # Download and load dataset
    benchmark.download_dataset()
    benchmark.load_dataset('s')  # Use small dataset for testing
    
    # Run benchmark on first 20 examples
    results = benchmark.run_benchmark(
        num_samples=20,
        start_idx=200,
        question_types=None,  # Test all types
        model="o3"
    )
    
    # Print results
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    print(f"Total Examples: {results['total_samples']}")
    print(f"Correct Answers: {results['correct_answers']}")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Average Duration: {results['avg_response_time']:.2f}s")
    print(f"Average Retrieval Duration: {results['avg_retrieval_duration']:.2f}s")
    
    print("\nBy Question Type:")
    for qtype, stats in results['type_statistics'].items():
        print(f"  {qtype}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.3f})")
    
    # Save results
    benchmark.save_results(results)


if __name__ == "__main__":
    main()