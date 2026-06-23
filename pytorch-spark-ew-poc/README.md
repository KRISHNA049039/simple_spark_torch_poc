# PyTorch Inferencing on Apache Spark - Electronic Warfare PoC

## Overview

This proof-of-concept demonstrates that **PyTorch model inference can be performed more efficiently on Apache Spark** by leveraging distributed computing, vectorized (Pandas) UDFs, and model broadcasting — applied to an **Electronic Warfare (EW) signal classification** scenario.

## Use Case: EW Signal Classification

In electronic warfare, real-time classification of intercepted radar and jammer signals is critical. This PoC simulates:
- **Radar pulse classification** (pulse type, modulation, threat level)
- **High-throughput inference** on millions of signal samples
- **Distributed processing** across Spark workers

## Project Structure

```
pytorch-spark-ew-poc/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── models/
│   └── ew_signal_model.py     # PyTorch signal classifier model
├── data/
│   └── generate_signals.py    # Synthetic EW signal data generator
├── inference/
│   ├── spark_inference.py     # Spark-distributed inference pipeline
│   └── baseline_inference.py  # Single-node sequential inference
├── benchmark/
│   └── run_benchmark.py       # Main benchmark runner with metrics
└── results/
    └── metrics_report.md      # Performance metrics and analysis
```

## Quick Start

```bash
pip install -r requirements.txt
python benchmark/run_benchmark.py
```

## Key Techniques

1. **Pandas UDFs (Vectorized UDFs)** — Batch inference via Arrow-optimized data transfer
2. **Model Broadcasting** — Serialize model once, broadcast to all workers
3. **Partition-level inference** — Process entire partitions with `mapInPandas`
4. **GPU-optional** — Automatically uses CUDA if available

## Expected Results

Spark-based inference shows significant speedup over sequential processing, particularly as data volume scales beyond what fits in single-node memory.
