#!/usr/bin/env python3
"""Stress test script for vLLM API endpoint.

Features:
- Multiple test patterns (sequential, parallel, burst, ramp, sustained)
- Uses varied documents to avoid prefix cache hits
- Validates output format (expects list of integers)
- Health checks and detailed statistics

Set VERBOSE=True to see each request's output in real-time.
Set VERBOSE=False to see only summary statistics and sample outputs.
"""

import sys
import time
import random
import ast
import json
import statistics
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import requests
from src.config import ChunkingConfig
from src.core.prompts import render_prediction_prompts

BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "qwen3_lora"  # Change to BASE_MODEL or LORA_MODEL as needed
VERBOSE = True  # Set to False to hide individual request outputs


@dataclass
class RequestResult:
    """Result of a single request."""
    success: bool
    latency: float  # seconds
    tokens_generated: Optional[int] = None
    error: Optional[str] = None
    output: Optional[str] = None


@dataclass
class TestStats:
    """Statistics for a test run."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    latencies: List[float] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        return (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
    
    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0
    
    @property
    def median_latency(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0
    
    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx]
    
    @property
    def p99_latency(self) -> float:
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[idx]
    
    @property
    def throughput(self) -> float:
        """Requests per second."""
        return self.successful_requests / self.total_time if self.total_time > 0 else 0
    
    def print_summary(self, test_name: str):
        """Print formatted test summary."""
        print(f"\n{'='*70}")
        print(f"Results: {test_name}")
        print(f"{'='*70}")
        print(f"Total Requests:      {self.total_requests}")
        print(f"Successful:          {self.successful_requests}")
        print(f"Failed:              {self.failed_requests}")
        print(f"Success Rate:        {self.success_rate:.2f}%")
        print(f"Total Time:          {self.total_time:.2f}s")
        print(f"Throughput:          {self.throughput:.2f} req/s")
        print(f"\nLatency Statistics:")
        print(f"  Average:           {self.avg_latency:.3f}s")
        print(f"  Median:            {self.median_latency:.3f}s")
        print(f"  P95:               {self.p95_latency:.3f}s")
        print(f"  P99:               {self.p99_latency:.3f}s")
        if self.latencies:
            print(f"  Min:               {min(self.latencies):.3f}s")
            print(f"  Max:               {max(self.latencies):.3f}s")
        print(f"{'='*70}\n")


def load_test_documents():
    """Load multiple test documents to avoid prefix cache hits."""
    docs = []
    
    # Try to load from training_pairs directory (these have proper <|loc_N|> markers)
    training_pairs_dir = project_root / "data" / "v1" / "training_pairs"
    if training_pairs_dir.exists():
        json_files = list(training_pairs_dir.glob("*.json"))[:100]  # Get up to 100 docs
        for f in json_files:
            try:
                data = json.loads(f.read_text())
                if "input" in data and data["input"]:
                    docs.append(data["input"])
            except:
                continue
    
    # If we still don't have enough docs, try the dummy data
    if len(docs) < 50:
        dummy_dir = project_root / "data" / "dummy" / "training_pairs"
        if dummy_dir.exists():
            json_files = list(dummy_dir.glob("*.json"))[:50]
            for f in json_files:
                try:
                    data = json.loads(f.read_text())
                    if "input" in data and data["input"]:
                        docs.append(data["input"])
                except:
                    continue
    
    # If we have less than 10 docs, add some synthetic variations as fallback
    if len(docs) < 10:
        base_topics = [
            ("Machine learning", "artificial intelligence", "neural networks", "training data", "model optimization"),
            ("Climate change", "global warming", "carbon emissions", "renewable energy", "sustainability"),
            ("Quantum computing", "qubits", "superposition", "entanglement", "quantum algorithms"),
            ("Blockchain technology", "cryptocurrencies", "distributed ledgers", "smart contracts", "decentralization"),
            ("Space exploration", "Mars missions", "satellite technology", "space stations", "astronomical research"),
            ("Biotechnology", "gene editing", "CRISPR", "protein synthesis", "medical applications"),
            ("Cybersecurity", "encryption", "network security", "threat detection", "data protection"),
            ("Renewable energy", "solar panels", "wind turbines", "energy storage", "grid integration"),
            ("Artificial intelligence", "machine learning models", "deep learning", "computer vision", "natural language processing"),
            ("Nanotechnology", "molecular engineering", "nanomaterials", "quantum dots", "medical nanodevices"),
        ]
        
        for topic, *keywords in base_topics:
            doc = f"<|loc_0|>{topic} is an emerging field of study. "
            for i, keyword in enumerate(keywords, 1):
                doc += f"<|loc_{i}|>Research in {keyword} has shown promising results. "
            docs.append(doc.strip())
    
    return docs


def make_request(doc_text: str, request_id: int, use_cache_bust: bool = True) -> RequestResult:
    """Make a single request to the API."""
    config = ChunkingConfig()
    
    # Optional: add a small random suffix to further bust cache
    if use_cache_bust:
        doc_text = doc_text + f" <|req_{request_id}|>"
    
    system_prompt, user_prompt = render_prediction_prompts(doc_text, config)
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 256,
                "temperature": 0.05,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=30
        )
        latency = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            output = data['choices'][0]['message']['content']
            tokens = data.get('usage', {}).get('completion_tokens')
            
            if VERBOSE:
                print(f"\n  [Request {request_id}] Latency: {latency:.3f}s | Tokens: {tokens}")
                print(f"    Output: {output[:150]}{'...' if len(output) > 150 else ''}")
            
            return RequestResult(success=True, latency=latency, tokens_generated=tokens, output=output)
        else:
            error_msg = f"HTTP {response.status_code}"
            if VERBOSE:
                print(f"\n  [Request {request_id}] ❌ {error_msg}")
            return RequestResult(success=False, latency=latency, error=error_msg)
    
    except Exception as e:
        latency = time.time() - start_time
        error_msg = str(e)
        if VERBOSE:
            print(f"\n  [Request {request_id}] ❌ {error_msg}")
        return RequestResult(success=False, latency=latency, error=error_msg)


def sequential_test(documents: List[str], num_requests: int = 10) -> TestStats:
    """Test with sequential requests."""
    print(f"\n[Sequential Test] Running {num_requests} sequential requests...")
    print(f"  Using {len(documents)} different documents (cycling)")
    
    results = []
    start_time = time.time()
    
    for i in range(num_requests):
        print(f"  Request {i+1}/{num_requests}...", end='\r')
        doc = documents[i % len(documents)]
        result = make_request(doc, i)
        results.append(result)
    
    total_time = time.time() - start_time
    print(f"  Completed {num_requests} requests in {total_time:.2f}s" + " "*20)
    
    return _compile_stats(results, total_time)


def parallel_test(documents: List[str], num_requests: int = 50, max_workers: int = 10) -> TestStats:
    """Test with parallel requests using ThreadPoolExecutor."""
    print(f"\n[Parallel Test] Running {num_requests} requests with {max_workers} workers...")
    print(f"  Using {len(documents)} different documents (random selection)")
    
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(make_request, random.choice(documents), i) for i in range(num_requests)]
        
        for i, future in enumerate(as_completed(futures), 1):
            print(f"  Completed {i}/{num_requests}...", end='\r')
            results.append(future.result())
    
    total_time = time.time() - start_time
    print(f"  Completed {num_requests} requests in {total_time:.2f}s" + " "*20)
    
    return _compile_stats(results, total_time)


def burst_test(documents: List[str], num_bursts: int = 5, requests_per_burst: int = 20, burst_delay: float = 2.0) -> TestStats:
    """Test with bursts of requests."""
    print(f"\n[Burst Test] Running {num_bursts} bursts of {requests_per_burst} requests (delay: {burst_delay}s)...")
    print(f"  Using {len(documents)} different documents (random selection)")
    
    all_results = []
    start_time = time.time()
    request_id = 0
    
    for burst_num in range(num_bursts):
        print(f"  Burst {burst_num+1}/{num_bursts}...")
        
        with ThreadPoolExecutor(max_workers=requests_per_burst) as executor:
            futures = [executor.submit(make_request, random.choice(documents), request_id + i) 
                      for i in range(requests_per_burst)]
            results = [future.result() for future in as_completed(futures)]
            all_results.extend(results)
            request_id += requests_per_burst
        
        if burst_num < num_bursts - 1:
            time.sleep(burst_delay)
    
    total_time = time.time() - start_time
    print(f"  Completed {num_bursts} bursts in {total_time:.2f}s" + " "*20)
    
    return _compile_stats(all_results, total_time)


def ramp_test(documents: List[str], start_workers: int = 1, max_workers: int = 20, step: int = 5, requests_per_step: int = 10) -> TestStats:
    """Test with ramping load."""
    print(f"\n[Ramp Test] Ramping from {start_workers} to {max_workers} workers (step: {step})...")
    print(f"  Using {len(documents)} different documents (random selection)")
    
    all_results = []
    start_time = time.time()
    request_id = 0
    
    current_workers = start_workers
    while current_workers <= max_workers:
        print(f"  Testing with {current_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=current_workers) as executor:
            futures = [executor.submit(make_request, random.choice(documents), request_id + i) 
                      for i in range(requests_per_step)]
            results = [future.result() for future in as_completed(futures)]
            all_results.extend(results)
            request_id += requests_per_step
        
        current_workers += step
    
    total_time = time.time() - start_time
    print(f"  Completed ramp test in {total_time:.2f}s" + " "*20)
    
    return _compile_stats(all_results, total_time)


def sustained_load_test(documents: List[str], duration_seconds: int = 30, target_rps: float = 5.0) -> TestStats:
    """Test with sustained load at target requests per second."""
    print(f"\n[Sustained Load Test] Running for {duration_seconds}s at {target_rps} req/s...")
    print(f"  Using {len(documents)} different documents (random selection)")
    
    results = []
    start_time = time.time()
    request_interval = 1.0 / target_rps
    next_request_time = start_time
    request_count = 0
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        
        while time.time() - start_time < duration_seconds:
            current_time = time.time()
            
            if current_time >= next_request_time:
                doc = random.choice(documents)
                future = executor.submit(make_request, doc, request_count)
                futures.append(future)
                request_count += 1
                next_request_time += request_interval
                
                print(f"  Requests sent: {request_count}, Elapsed: {current_time - start_time:.1f}s", end='\r')
            else:
                time.sleep(0.01)  # Small sleep to avoid busy waiting
        
        # Wait for all requests to complete
        print(f"\n  Waiting for {len(futures)} requests to complete...")
        for future in as_completed(futures):
            results.append(future.result())
    
    total_time = time.time() - start_time
    print(f"  Completed {len(results)} requests in {total_time:.2f}s" + " "*20)
    
    return _compile_stats(results, total_time)


def validate_output(output: str) -> Tuple[bool, Optional[List[int]]]:
    """Validate that output is a valid list of integers."""
    try:
        parsed = ast.literal_eval(output.strip())
        if isinstance(parsed, list) and all(isinstance(x, int) for x in parsed):
            return True, parsed
        return False, None
    except:
        return False, None


def _compile_stats(results: List[RequestResult], total_time: float) -> TestStats:
    """Compile results into statistics."""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    stats = TestStats(
        total_requests=len(results),
        successful_requests=len(successful),
        failed_requests=len(failed),
        total_time=total_time,
        latencies=[r.latency for r in successful]
    )
    
    # Health check: validate outputs
    if successful:
        valid_outputs = 0
        invalid_outputs = []
        
        for i, result in enumerate(successful):
            if result.output:
                is_valid, parsed = validate_output(result.output)
                if is_valid:
                    valid_outputs += 1
                else:
                    invalid_outputs.append((i, result.output))
        
        print(f"\n  Health Check:")
        print(f"    Valid outputs: {valid_outputs}/{len(successful)} ({valid_outputs/len(successful)*100:.1f}%)")
        
        if invalid_outputs:
            print(f"    ⚠️  Invalid outputs detected:")
            for idx, output in invalid_outputs[:3]:  # Show first 3 invalid
                print(f"      [{idx}] {output[:80]}...")
        
        # Show sample valid outputs if not in verbose mode
        if not VERBOSE:
            print(f"\n  Sample outputs (first 3):")
            count = 0
            for result in successful:
                if result.output and count < 3:
                    is_valid, parsed = validate_output(result.output)
                    if is_valid:
                        print(f"    ✓ {result.output}")
                        count += 1
    
    return stats


def check_server():
    """Check if server is available."""
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    """Run all stress tests."""
    print("="*70)
    print("vLLM API Stress Test")
    print("="*70)
    print(f"Target URL: {BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print()
    
    # Check server availability
    print("Checking server availability...", end=' ')
    if not check_server():
        print("❌")
        print("\n⚠️  Server not available. Please start the vLLM server first.")
        print("Run: bash scripts/start_vllm_server.sh")
        sys.exit(1)
    print("✅")
    
    # Load test documents
    print("\nLoading test documents...", end=' ')
    documents = load_test_documents()
    print(f"✅ Loaded {len(documents)} documents")
    if documents:
        avg_len = sum(len(d) for d in documents) / len(documents)
        print(f"  Average document length: {avg_len:.0f} characters")
    
    # Run tests
    tests = []
    
    # 1. Sequential baseline
    stats = sequential_test(documents, num_requests=10)
    stats.print_summary("Sequential Test (10 requests)")
    tests.append(("Sequential", stats))
    
    # 2. Parallel test
    stats = parallel_test(documents, num_requests=50, max_workers=10)
    stats.print_summary("Parallel Test (50 requests, 10 workers)")
    tests.append(("Parallel", stats))
    
    # 3. Burst test
    stats = burst_test(documents, num_bursts=5, requests_per_burst=10, burst_delay=2.0)
    stats.print_summary("Burst Test (5 bursts × 10 requests)")
    tests.append(("Burst", stats))
    
    # 4. Ramp test
    stats = ramp_test(documents, start_workers=1, max_workers=15, step=5, requests_per_step=10)
    stats.print_summary("Ramp Test (1→15 workers, step 5)")
    tests.append(("Ramp", stats))
    
    # 5. Sustained load test
    stats = sustained_load_test(documents, duration_seconds=30, target_rps=3.0)
    stats.print_summary("Sustained Load Test (30s @ 3 req/s)")
    tests.append(("Sustained", stats))
    
    # Overall summary
    print("\n" + "="*70)
    print("Overall Summary")
    print("="*70)
    print(f"{'Test':<20} {'Requests':<12} {'Success':<12} {'Avg Latency':<15} {'Throughput'}")
    print("-"*70)
    for name, stats in tests:
        print(f"{name:<20} {stats.total_requests:<12} {stats.success_rate:>6.1f}%     {stats.avg_latency:>8.3f}s       {stats.throughput:>6.2f} req/s")
    print("="*70)


if __name__ == "__main__":
    main()

