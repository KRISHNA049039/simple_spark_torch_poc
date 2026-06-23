"""
Baseline Single-Node PyTorch Inference for EW Signal Classification

This module implements sequential (non-distributed) inference approaches to
serve as the performance baseline against Spark-distributed inference.

Strategies implemented:
1. Single-sample inference (worst case - no batching)
2. Batched inference (standard PyTorch best practice on single node)
3. DataLoader-based inference (PyTorch DataLoader with num_workers)

These baselines represent what you'd typically do WITHOUT Spark, showing
the scaling limitations of single-node processing for large signal datasets.
"""

import sys
import os
import time
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ew_signal_model import (
    create_trained_model, SIGNAL_CLASSES, INPUT_FEATURES
)
from data.generate_signals import generate_signal_dataset


def _get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_single_sample_inference(model, features, labels):
    """
    Strategy 1: Single-sample sequential inference (worst case).

    Processes one signal sample at a time - represents naive implementation
    without any batching optimization.

    Args:
        model: Trained PyTorch model in eval mode
        features: numpy array (N, 128)
        labels: numpy array (N,)

    Returns:
        Tuple of (predictions, elapsed_time, throughput, latency_per_sample)
    """
    model.eval()
    num_samples = len(features)
    predictions = np.zeros(num_samples, dtype=np.int64)
    latencies = []
    device = _get_device()
    model = model.to(device)

    start_time = time.time()

    with torch.no_grad():
        for i in range(num_samples):
            sample_start = time.time()
            x = torch.FloatTensor(features[i:i+1]).to(device)
            pred = model(x)
            predictions[i] = torch.argmax(pred, dim=1).item()
            latencies.append(time.time() - sample_start)

    elapsed_time = time.time() - start_time
    throughput = num_samples / elapsed_time
    avg_latency = np.mean(latencies) * 1000  # Convert to ms
    p99_latency = np.percentile(latencies, 99) * 1000

    return predictions, elapsed_time, throughput, avg_latency, p99_latency


def run_batched_inference(model, features, labels, batch_size=1024):
    """
    Strategy 2: Batched inference on single node.

    Standard PyTorch practice - processes signals in fixed-size batches.
    Better than single-sample but still limited to one machine.

    Args:
        model: Trained PyTorch model in eval mode
        features: numpy array (N, 128)
        labels: numpy array (N,)
        batch_size: Number of samples per batch

    Returns:
        Tuple of (predictions, elapsed_time, throughput, avg_latency_ms, p99_latency_ms)
    """
    model.eval()
    num_samples = len(features)
    predictions = np.zeros(num_samples, dtype=np.int64)
    batch_latencies = []
    device = _get_device()
    model = model.to(device)

    start_time = time.time()

    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            batch_start = time.time()
            end = min(start + batch_size, num_samples)
            x = torch.FloatTensor(features[start:end]).to(device)
            pred = model(x)
            predictions[start:end] = torch.argmax(pred, dim=1).cpu().numpy()
            batch_latencies.append(time.time() - batch_start)

    elapsed_time = time.time() - start_time
    throughput = num_samples / elapsed_time

    # Per-sample latency derived from batch latency
    avg_latency = np.mean(batch_latencies) / batch_size * 1000
    p99_latency = np.percentile(batch_latencies, 99) / batch_size * 1000

    return predictions, elapsed_time, throughput, avg_latency, p99_latency


def run_dataloader_inference(model, features, labels, batch_size=1024, num_workers=0):
    """
    Strategy 3: PyTorch DataLoader-based inference.

    Uses DataLoader for data loading parallelism (pre-fetching, multi-worker).
    Represents the best single-node approach with PyTorch built-in utilities.

    Args:
        model: Trained PyTorch model in eval mode
        features: numpy array (N, 128)
        labels: numpy array (N,)
        batch_size: DataLoader batch size
        num_workers: Number of data loading workers

    Returns:
        Tuple of (predictions, elapsed_time, throughput, avg_latency_ms, p99_latency_ms)
    """
    model.eval()
    num_samples = len(features)
    device = _get_device()
    model = model.to(device)

    dataset = TensorDataset(
        torch.FloatTensor(features),
        torch.LongTensor(labels)
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    all_predictions = []
    batch_latencies = []

    start_time = time.time()

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_start = time.time()
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            preds = torch.argmax(pred, dim=1)
            all_predictions.append(preds.cpu().numpy())
            batch_latencies.append(time.time() - batch_start)

    elapsed_time = time.time() - start_time
    predictions = np.concatenate(all_predictions)
    throughput = num_samples / elapsed_time

    avg_latency = np.mean(batch_latencies) / batch_size * 1000
    p99_latency = np.percentile(batch_latencies, 99) / batch_size * 1000

    return predictions, elapsed_time, throughput, avg_latency, p99_latency


def run_full_baseline_benchmark(num_samples=100000, batch_size=1024):
    """
    Run all baseline inference strategies and return metrics.

    Args:
        num_samples: Number of signal samples
        batch_size: Batch size for batched strategies

    Returns:
        Dict with timing and throughput for each baseline strategy
    """
    print(f"\n{'='*60}")
    print(f"BASELINE SINGLE-NODE INFERENCE BENCHMARK")
    print(f"Samples: {num_samples:,} | Batch Size: {batch_size}")
    print(f"{'='*60}")

    # Setup
    print("\n[1/3] Training model...")
    model = create_trained_model()

    print("[2/3] Generating signal data...")
    features, labels = generate_signal_dataset(num_samples)

    print("[3/3] Running baseline strategies...\n")

    results = {}

    # For single-sample, cap at 10K samples to avoid excessive runtime
    single_sample_count = min(num_samples, 10000)

    # Strategy 1: Single sample (capped)
    print(f"  -> Single-sample inference ({single_sample_count:,} samples)...", end=" ", flush=True)
    _, time_single, tp_single, lat_single, p99_single = run_single_sample_inference(
        model, features[:single_sample_count], labels[:single_sample_count]
    )
    results["single_sample"] = {
        "time": time_single,
        "throughput": tp_single,
        "avg_latency_ms": lat_single,
        "p99_latency_ms": p99_single,
        "num_samples": single_sample_count,
        "extrapolated_time": time_single * (num_samples / single_sample_count),
    }
    print(f"{time_single:.2f}s ({tp_single:,.0f} samples/sec)")

    # Strategy 2: Batched inference
    print(f"  -> Batched inference (batch={batch_size})...", end=" ", flush=True)
    preds_batch, time_batch, tp_batch, lat_batch, p99_batch = run_batched_inference(
        model, features, labels, batch_size
    )
    accuracy = (preds_batch == labels).mean()
    results["batched"] = {
        "time": time_batch,
        "throughput": tp_batch,
        "avg_latency_ms": lat_batch,
        "p99_latency_ms": p99_batch,
        "accuracy": accuracy,
        "num_samples": num_samples,
    }
    print(f"{time_batch:.2f}s ({tp_batch:,.0f} samples/sec) | Accuracy: {accuracy:.2%}")

    # Strategy 3: DataLoader inference
    print(f"  -> DataLoader inference (batch={batch_size}, workers=0)...", end=" ", flush=True)
    _, time_dl, tp_dl, lat_dl, p99_dl = run_dataloader_inference(
        model, features, labels, batch_size, num_workers=0
    )
    results["dataloader"] = {
        "time": time_dl,
        "throughput": tp_dl,
        "avg_latency_ms": lat_dl,
        "p99_latency_ms": p99_dl,
        "num_samples": num_samples,
    }
    print(f"{time_dl:.2f}s ({tp_dl:,.0f} samples/sec)")

    return results


if __name__ == "__main__":
    results = run_full_baseline_benchmark(num_samples=50000)
    print("\n--- Baseline Summary ---")
    for strategy, metrics in results.items():
        print(f"  {strategy}: {metrics['time']:.2f}s | {metrics['throughput']:,.0f} samples/sec")
