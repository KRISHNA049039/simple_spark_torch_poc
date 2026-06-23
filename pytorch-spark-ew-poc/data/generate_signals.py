"""
Synthetic EW Signal Data Generator

Generates large-scale synthetic electronic warfare signal datasets for
benchmarking inference throughput on Spark vs single-node processing.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ew_signal_model import _generate_signal_features, NUM_CLASSES, INPUT_FEATURES


def generate_signal_dataset(num_samples, seed=42):
    """
    Generate a large dataset of synthetic EW signal feature vectors.

    Args:
        num_samples: Total number of signal samples to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (features_array, labels_array)
        - features_array: shape (num_samples, 128) float32
        - labels_array: shape (num_samples,) int64
    """
    np.random.seed(seed)

    features = np.zeros((num_samples, INPUT_FEATURES), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.int64)

    for i in range(num_samples):
        class_idx = np.random.randint(0, NUM_CLASSES)
        features[i] = _generate_signal_features(class_idx)
        labels[i] = class_idx

    return features, labels


def generate_signal_dataset_chunked(num_samples, chunk_size=10000, seed=42):
    """
    Generator that yields signal data in chunks (memory-efficient for large datasets).

    Args:
        num_samples: Total number of signal samples
        chunk_size: Samples per chunk
        seed: Random seed

    Yields:
        Tuple of (features_chunk, labels_chunk)
    """
    np.random.seed(seed)

    num_chunks = (num_samples + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, num_samples)
        actual_size = end - start

        features = np.zeros((actual_size, INPUT_FEATURES), dtype=np.float32)
        labels = np.zeros(actual_size, dtype=np.int64)

        for i in range(actual_size):
            class_idx = np.random.randint(0, NUM_CLASSES)
            features[i] = _generate_signal_features(class_idx)
            labels[i] = class_idx

        yield features, labels


if __name__ == "__main__":
    print("Generating sample EW signal dataset...")
    features, labels = generate_signal_dataset(1000)
    print(f"Generated {len(features)} samples")
    print(f"Feature shape: {features.shape}")
    print(f"Label distribution: {np.bincount(labels)}")
    print(f"Memory usage: {features.nbytes / 1024:.1f} KB")
