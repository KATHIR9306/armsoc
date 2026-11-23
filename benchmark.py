"""
Comprehensive Benchmarking Suite for Industrial QC Assistant.
Measures Vision Pipeline Latencies, Throughput (FPS), and Llama+ExecuTorch Metrics.
Directly addresses Slide 7 & 11 Evaluation Criteria.
"""

import time
import os
import sys
import tracemalloc
import numpy as np
import pandas as pd
from typing import Dict, Any

# Ensure project root in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.assistant import QualityControlAssistant

def run_benchmarks(num_iterations: int = 50) -> Dict[str, Any]:
    print(f"\n========================================================")
    print(f"🏭 BHARAT AI-SOC CHALLENGE: EDGE BENCHMARK SUITE (PS5)")
    print(f"========================================================")
    print(f"Running {num_iterations} edge inference iterations...\n")

    tracemalloc.start()
    qc = QualityControlAssistant(db_path="data/benchmark_test.db")

    vision_pre_times = []
    vision_infer_times = []
    vision_post_times = []
    vision_total_times = []

    llama_ttft_times = []
    llama_tok_speeds = []
    llama_total_times = []

    # Warmup
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    for _ in range(5):
        qc.inspect_unit(dummy_frame)

    # Benchmark loop
    start_total = time.perf_counter()

    for i in range(num_iterations):
        res = qc.inspect_unit(dummy_frame, unit_id=f"BENCH-{i+1:04d}")
        f = res["finding"]
        a = res["advisory"]

        vision_pre_times.append(f["pre_process_ms"])
        vision_infer_times.append(f["inference_ms"])
        vision_post_times.append(f["post_process_ms"])
        vision_total_times.append(f["total_vision_latency_ms"])

        if a["time_to_first_token_ms"] > 0:
            llama_ttft_times.append(a["time_to_first_token_ms"])
            llama_tok_speeds.append(a["tokens_per_second"])
            llama_total_times.append(a["generation_time_ms"])

    total_time_sec = time.perf_counter() - start_total
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak_mem / (1024 * 1024)

    # Aggregate statistics
    v_mean_pre = np.mean(vision_pre_times)
    v_mean_inf = np.mean(vision_infer_times)
    v_mean_post = np.mean(vision_post_times)
    v_mean_total = np.mean(vision_total_times)
    v_fps = 1000.0 / v_mean_total if v_mean_total > 0 else 0

    l_mean_ttft = np.mean(llama_ttft_times) if llama_ttft_times else 0
    l_mean_speed = np.mean(llama_tok_speeds) if llama_tok_speeds else 0
    l_mean_total = np.mean(llama_total_times) if llama_total_times else 0

    results = {
        "iterations": num_iterations,
        "vision_pre_process_ms": round(float(v_mean_pre), 2),
        "vision_inference_ms": round(float(v_mean_inf), 2),
        "vision_post_process_ms": round(float(v_mean_post), 2),
        "vision_total_latency_ms": round(float(v_mean_total), 2),
        "vision_throughput_fps": round(float(v_fps), 1),
        "llama_time_to_first_token_ms": round(float(l_mean_ttft), 2),
        "llama_generation_speed_tok_s": round(float(l_mean_speed), 2),
        "llama_total_response_time_ms": round(float(l_mean_total), 2),
        "peak_memory_mb": round(float(peak_memory_mb), 2)
    }

    print("📊 BENCHMARK RESULTS SUMMARY:")
    print("--------------------------------------------------------")
    print(f"• Vision Pre-processing Latency : {results['vision_pre_process_ms']} ms")
    print(f"• Vision Model Inference Latency: {results['vision_inference_ms']} ms")
    print(f"• Vision Post-processing Latency: {results['vision_post_process_ms']} ms")
    print(f"• Total Vision Pipeline Latency : {results['vision_total_latency_ms']} ms")
    print(f"• Vision Line-Rate Throughput   : {results['vision_throughput_fps']} FPS")
    print("--------------------------------------------------------")
    print(f"• Llama Time to 1st Token (TTFT): {results['llama_time_to_first_token_ms']} ms")
    print(f"• Llama Generation Speed        : {results['llama_generation_speed_tok_s']} tokens/sec")
    print(f"• Llama Total Advisory Time     : {results['llama_total_response_time_ms']} ms")
    print(f"• Process Peak RAM Footprint    : {results['peak_memory_mb']} MB")
    print("========================================================\n")

    # Clean up benchmark temp db
    if os.path.exists("data/benchmark_test.db"):
        os.remove("data/benchmark_test.db")

    return results

if __name__ == "__main__":
    run_benchmarks(30)
