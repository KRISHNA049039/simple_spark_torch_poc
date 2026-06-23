# Airgapped Cluster Deployment Guide

## Overview

This guide covers deploying the PyTorch-on-Spark EW inference pipeline to an airgapped (no internet) cluster with GPU support.

## Prerequisites on Airgapped Cluster

- Spark cluster (Standalone, YARN, or K8s) 
- NVIDIA GPU drivers installed on worker nodes
- Docker runtime with nvidia-container-toolkit (if using containers)
- OR: Python 3.11+, Java 17+, NVIDIA CUDA toolkit on bare metal

---

## Option A: Docker-Based Deployment (Recommended)

### Step 1: Build Image (on internet-connected machine)

```bash
cd pytorch-spark-ew-poc
docker build -t ew-pytorch-spark:latest .
```

### Step 2: Export Image

**Linux / macOS:**
```bash
docker save ew-pytorch-spark:latest | gzip > ew-pytorch-spark.tar.gz
# Image size: ~6-8 GB (includes CUDA runtime + PyTorch + PySpark)
```

**Windows (CMD or PowerShell):**
```cmd
:: Tag the compose-built image with a clean name
docker tag pytorch-spark-ew-poc-benchmark:latest ew-pytorch-spark:latest

:: Export to tar file (no gzip pipe on Windows)
docker save ew-pytorch-spark:latest -o ew-pytorch-spark.tar

:: Optional: compress with 7-Zip to reduce transfer size
7z a ew-pytorch-spark.tar.gz ew-pytorch-spark.tar
```

> **Windows Note:** The `|` pipe with `gzip` does not work in CMD/PowerShell. Use the `-o` flag instead. The uncompressed `.tar` file (~6-8 GB) can be loaded directly on the target — compression is optional but recommended for USB transfer.

### Step 3: Transfer to Airgapped Environment

Transfer `ew-pytorch-spark.tar.gz` via:
- Encrypted USB drive
- Cross-domain file transfer system
- Approved data diode

### Step 4: Load on All Cluster Nodes

**Linux:**
```bash
docker load < ew-pytorch-spark.tar.gz
# or if uncompressed:
docker load -i ew-pytorch-spark.tar
```

**Windows:**
```cmd
docker load -i ew-pytorch-spark.tar
```

### Step 5: Run with GPU

```bash
# Single node test
docker run --gpus all -v /data/signals:/app/data ew-pytorch-spark:latest

# Or with docker-compose on cluster
docker run --gpus all \
  -e SPARK_MASTER=spark://master-node:7077 \
  -v /data/signals:/app/data \
  -v /output:/app/results \
  ew-pytorch-spark:latest \
  python benchmark/run_benchmark.py
```

---

## Option B: Bare Metal / spark-submit

### Step 1: Package Dependencies (on internet-connected machine)

```bash
# Create offline wheel archive
mkdir -p offline_packages
pip download torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121 -d offline_packages/
pip download pyspark==3.5.1 numpy pandas pyarrow -d offline_packages/
tar -czf ew_poc_packages.tar.gz offline_packages/
```

### Step 2: Transfer Files

Transfer to airgapped cluster:
- `ew_poc_packages.tar.gz` (Python wheels)
- `pytorch-spark-ew-poc/` (source code)

### Step 3: Install on Cluster

```bash
# On each node (or shared filesystem)
tar -xzf ew_poc_packages.tar.gz
pip install --no-index --find-links=offline_packages/ torch pyspark numpy pandas pyarrow
```

### Step 4: Run via spark-submit

```bash
spark-submit \
  --master spark://master:7077 \
  --deploy-mode client \
  --num-executors 8 \
  --executor-cores 4 \
  --executor-memory 8g \
  --driver-memory 4g \
  --conf spark.executor.resource.gpu.amount=1 \
  --conf spark.task.resource.gpu.amount=0.5 \
  --conf spark.network.timeout=600s \
  --conf spark.executor.heartbeatInterval=120s \
  benchmark/run_benchmark.py
```

---

## Option C: Kubernetes / Spark on K8s

### Step 1: Push Image to Internal Registry

```bash
docker tag ew-pytorch-spark:latest internal-registry.mil/ew/pytorch-spark:latest
docker push internal-registry.mil/ew/pytorch-spark:latest
```

### Step 2: Submit via K8s

```bash
spark-submit \
  --master k8s://https://k8s-master:6443 \
  --deploy-mode cluster \
  --conf spark.kubernetes.container.image=internal-registry.mil/ew/pytorch-spark:latest \
  --conf spark.kubernetes.executor.request.cores=4 \
  --conf spark.kubernetes.executor.limit.cores=4 \
  --conf spark.executor.memory=8g \
  --conf spark.executor.resource.gpu.amount=1 \
  --conf spark.kubernetes.executor.resources.nvidia.com/gpu.amount=1 \
  local:///app/benchmark/run_benchmark.py
```

---

## Cluster Configuration for Production EW

### Sizing Guidelines

| Signal Ingest Rate | Recommended Cluster | Expected Throughput |
|-------------------|--------------------|--------------------|
| 10K signals/sec | 2 nodes, 1 GPU each | ~50K inferences/sec |
| 100K signals/sec | 4 nodes, 2 GPUs each | ~400K inferences/sec |
| 1M signals/sec | 8 nodes, 4 GPUs each | ~2M inferences/sec |

### Spark Config for Low-Latency EW

```properties
# spark-defaults.conf
spark.executor.instances=8
spark.executor.cores=4
spark.executor.memory=8g
spark.executor.resource.gpu.amount=1
spark.task.resource.gpu.amount=0.5
spark.network.timeout=600s
spark.locality.wait=0s
spark.scheduler.mode=FAIR
spark.speculation=true
spark.speculation.interval=100ms
```

### Real-Time Streaming Integration

For continuous signal processing, wrap inference in Structured Streaming:

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf

# Read from Kafka / socket / file stream
signals_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "broker:9092") \
    .option("subscribe", "ew_signals") \
    .load()

# Apply inference
classified = signals_stream.foreachBatch(
    lambda batch_df, batch_id: run_inference_on_batch(batch_df, model)
)

classified.start().awaitTermination()
```

---

## Security Considerations for Airgapped EW Systems

- All model weights remain on-premise (no external model registry calls)
- No telemetry or external network calls in inference code
- Model serialization uses standard PyTorch state_dict (auditable)
- Spark communication can be encrypted with TLS between nodes
- Signal data never leaves the cluster boundary

---

## Verification Checklist

- [ ] Docker image loads on all nodes: `docker images | grep ew-pytorch`
- [ ] GPU visible inside container: `docker run --gpus all ew-pytorch-spark nvidia-smi`
- [ ] Spark cluster healthy: `spark-submit --master spark://master:7077 --deploy-mode client test_spark.py`
- [ ] Inference produces correct accuracy (>90% on synthetic data)
- [ ] Throughput meets mission requirements at target signal rate
