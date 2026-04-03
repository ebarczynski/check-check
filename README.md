# PyTorch RGCN Hardware Benchmark

This project benchmarks the same PyTorch Relational Graph Convolutional Network workload across:

- NVIDIA GPUs through the PyTorch CUDA backend
- AMD GPUs through the PyTorch ROCm backend
- Apple Silicon through the PyTorch MPS backend (Metal)

The benchmark is intentionally designed to keep accelerators busy for long runs. The default presets use large synthetic relational graphs, multiple RGCN layers, and an MLP inside each block. Every preset also enforces a minimum training duration so runs do not end before one hour unless you explicitly override that behavior.

## Why MPS Instead Of MLX

MLX is a separate framework. If you want a fair backend comparison while staying inside the PyTorch framework, the Apple target should be PyTorch MPS, which uses Metal underneath.

If you also want an MLX-native RGCN implementation, that should be treated as a second benchmark track rather than mixed into this one, because it changes both the framework and operator stack.

## Benchmark Design

- Uses a synthetic full-batch relational graph, so every platform runs the same deterministic workload without dataset download friction.
- Keeps the model backend-portable by implementing RGCN message passing directly in PyTorch with dense relation weights and `index_add_` aggregation.
- Adds a feed-forward expansion inside each RGCN block so the run is not purely scatter-bound.
- Stops only after both conditions are satisfied: target epoch count reached and minimum wall-clock duration reached.
- Writes a per-epoch CSV log plus a machine-readable JSON summary for cross-platform comparison.

## Install

1. Install a PyTorch build that matches the hardware backend on the target machine.
2. Install this package in editable mode.

Examples:

```bash
python -m pip install -U pip
python -m pip install -e .
```

Notes:

- For NVIDIA CUDA and AMD ROCm, install PyTorch from the current official PyTorch selector for the exact driver and runtime version on that machine.
- Apple Silicon wheels from the standard PyPI flow expose the `mps` backend when macOS and PyTorch support it.

## Quick Smoke Test

Use the small config first to confirm the backend works end-to-end before starting a 1h+ run.

```bash
python -m rgcn_benchmark.train --config configs/smoke_test.json --device auto
```

## Long-Run Presets

24 GB class NVIDIA or AMD GPUs:

```bash
python -m rgcn_benchmark.train --config configs/cuda_rocm_24gb_long.json --device cuda
```

48 GB class NVIDIA or AMD GPUs:

```bash
python -m rgcn_benchmark.train --config configs/cuda_rocm_48gb_stress.json --device cuda
```

Apple Silicon with larger unified memory:

```bash
python -m rgcn_benchmark.train --config configs/apple_mps_64gb_long.json --device mps
```

Important:

- PyTorch on ROCm still uses `cuda` as the device type, so the ROCm command also uses `--device cuda`.
- The benchmark keeps training until both `epochs` and `min_duration_sec` are satisfied.
- If you want a fixed 1h floor everywhere, keep `min_duration_sec` at `3600`.

## Useful Overrides

Increase pressure on faster hardware:

```bash
python -m rgcn_benchmark.train \
  --config configs/cuda_rocm_48gb_stress.json \
  --device cuda \
  --hidden-dim 1280 \
  --num-layers 6 \
  --edges-per-relation 1800000
```

Reduce memory pressure after an out-of-memory failure:

```bash
python -m rgcn_benchmark.train \
  --config configs/apple_mps_64gb_long.json \
  --device mps \
  --hidden-dim 512 \
  --edge-chunk-size 8192 \
  --num-nodes 80000
```

Change precision when the backend supports it:

```bash
python -m rgcn_benchmark.train \
  --config configs/cuda_rocm_24gb_long.json \
  --device cuda \
  --precision fp16
```

## Results

Each run creates a timestamped directory under `results/` with:

- `summary.json`: backend, device, workload, config, and aggregate metrics
- `history.csv`: per-epoch loss, accuracy, duration, throughput, and memory statistics

Compare multiple completed runs:

```bash
python -m rgcn_benchmark.compare \
  results/20260403T100000Z_cuda24_cuda/summary.json \
  results/20260403T120000Z_rocm24_rocm/summary.json \
  results/20260403T140000Z_mps64_mps/summary.json
```

## Tuning Guidance

- If GPU utilization is low, increase `hidden_dim`, `num_layers`, or `edges_per_relation`.
- If memory usage is the limiter, lower `hidden_dim` before lowering `num_relations`.
- If the run is too short, raise `min_duration_sec` or scale the graph up.
- On Apple MPS, smaller `edge_chunk_size` values are often more stable than aggressive chunk sizes.
