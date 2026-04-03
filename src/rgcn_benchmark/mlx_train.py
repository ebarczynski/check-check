from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import shlex
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from rgcn_benchmark import benchmark_utils

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ModuleNotFoundError:
    mx = None
    nn = None
    optim = None


@dataclass
class GraphConfig:
    num_nodes: int = 120_000
    num_relations: int = 12
    edges_per_relation: int = 1_200_000
    input_dim: int = 256
    num_classes: int = 64
    train_fraction: float = 0.8
    seed: int = 17


@dataclass
class ModelConfig:
    hidden_dim: int = 768
    num_layers: int = 4
    dropout: float = 0.1
    edge_chunk_size: int = 32_768
    ffn_multiplier: int = 2


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 7e-4
    weight_decay: float = 1e-2


@dataclass
class RunConfig:
    epochs: int = 400
    min_duration_sec: float = 3600.0
    warmup_epochs: int = 5
    log_every: int = 1
    precision: str = "fp32"
    results_dir: str = "results"
    tf32: bool = False


@dataclass
class BenchmarkConfig:
    name: str = "apple_mlx_long"
    description: str = "MLX-native Apple RGCN benchmark"
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    run: RunConfig = field(default_factory=RunConfig)
    sizing: benchmark_utils.SizingConfig = field(
        default_factory=benchmark_utils.SizingConfig
    )


@dataclass
class SyntheticGraph:
    features: Any
    labels: Any
    train_index: Any
    train_labels: Any
    relations: list[tuple[Any, Any, Any]]
    total_edges: int


def ensure_mlx_available() -> None:
    if mx is None or nn is None or optim is None:
        raise RuntimeError(
            "MLX is not installed. On Apple Silicon, install it with `python -m pip install -e .[apple]`."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark an MLX-native RGCN workload on Apple Silicon."
    )
    parser.add_argument("--config", type=Path, help="Path to a JSON config preset.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "gpu", "cpu"),
        help="Target MLX device.",
    )
    parser.add_argument("--name", help="Override the run name.")
    parser.add_argument("--description", help="Override the run description.")
    parser.add_argument("--results-dir", type=Path, help="Override the results directory.")
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        help="Override math precision.",
    )
    parser.add_argument("--epochs", type=int, help="Override target epoch count.")
    parser.add_argument(
        "--min-duration-sec",
        type=float,
        help="Override minimum benchmark duration in seconds.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        help="Exclude the first N epochs from steady-state summary metrics.",
    )
    parser.add_argument(
        "--auto-size",
        action="store_true",
        help="Probe unified-memory headroom and scale the graph toward a target budget before the full run.",
    )
    parser.add_argument(
        "--target-memory-gb",
        type=float,
        help="Absolute unified-memory budget to target during autosizing.",
    )
    parser.add_argument(
        "--target-memory-fraction",
        type=float,
        help="Fraction of available unified memory to target during autosizing.",
    )
    parser.add_argument(
        "--sizing-tolerance",
        type=float,
        help="Relative autosizing tolerance, for example 0.08 for +/-8%%.",
    )
    parser.add_argument(
        "--sizing-max-probes",
        type=int,
        help="Maximum number of autosizing calibration probes.",
    )
    parser.add_argument("--num-nodes", type=int, help="Override number of nodes.")
    parser.add_argument("--num-relations", type=int, help="Override number of edge relations.")
    parser.add_argument("--edges-per-relation", type=int, help="Override edges per relation.")
    parser.add_argument("--input-dim", type=int, help="Override input feature width.")
    parser.add_argument("--num-classes", type=int, help="Override class count.")
    parser.add_argument(
        "--train-fraction",
        type=float,
        help="Override fraction of nodes used for the training loss.",
    )
    parser.add_argument("--seed", type=int, help="Override random seed.")
    parser.add_argument("--hidden-dim", type=int, help="Override hidden dimension.")
    parser.add_argument("--num-layers", type=int, help="Override number of RGCN blocks.")
    parser.add_argument("--dropout", type=float, help="Override dropout probability.")
    parser.add_argument(
        "--edge-chunk-size",
        type=int,
        help="Override edge chunk size for relation aggregation.",
    )
    parser.add_argument(
        "--ffn-multiplier",
        type=int,
        help="Override the hidden expansion factor inside each RGCN block.",
    )
    parser.add_argument("--lr", type=float, help="Override learning rate.")
    parser.add_argument("--weight-decay", type=float, help="Override optimizer weight decay.")
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "sgd"),
        help="Override optimizer type.",
    )
    return parser.parse_args()


def load_benchmark_config(path: Path | None) -> BenchmarkConfig:
    config = BenchmarkConfig()
    if path is None:
        return config

    resolved_path = benchmark_utils.resolve_config_path(path)
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    config.name = payload.get("name", config.name)
    config.description = payload.get("description", config.description)

    if "graph" in payload:
        config.graph = replace(config.graph, **payload["graph"])
    if "model" in payload:
        config.model = replace(config.model, **payload["model"])
    if "optimizer" in payload:
        config.optimizer = replace(config.optimizer, **payload["optimizer"])
    if "run" in payload:
        config.run = replace(config.run, **payload["run"])
    if "sizing" in payload:
        config.sizing = replace(config.sizing, **payload["sizing"])

    return config


def apply_cli_overrides(config: BenchmarkConfig, args: argparse.Namespace) -> BenchmarkConfig:
    if args.name:
        config.name = args.name
    if args.description:
        config.description = args.description
    if args.results_dir:
        config.run.results_dir = str(args.results_dir)
    if args.precision:
        config.run.precision = args.precision
    if args.epochs is not None:
        config.run.epochs = args.epochs
    if args.min_duration_sec is not None:
        config.run.min_duration_sec = args.min_duration_sec
    if args.warmup_epochs is not None:
        config.run.warmup_epochs = args.warmup_epochs
    if args.auto_size:
        config.sizing.enabled = True
    if args.target_memory_gb is not None:
        config.sizing.enabled = True
        config.sizing.target_memory_gb = args.target_memory_gb
    if args.target_memory_fraction is not None:
        config.sizing.enabled = True
        config.sizing.target_memory_fraction = args.target_memory_fraction
    if args.sizing_tolerance is not None:
        config.sizing.enabled = True
        config.sizing.tolerance = args.sizing_tolerance
    if args.sizing_max_probes is not None:
        config.sizing.enabled = True
        config.sizing.max_probes = args.sizing_max_probes
    if args.num_nodes is not None:
        config.graph.num_nodes = args.num_nodes
    if args.num_relations is not None:
        config.graph.num_relations = args.num_relations
    if args.edges_per_relation is not None:
        config.graph.edges_per_relation = args.edges_per_relation
    if args.input_dim is not None:
        config.graph.input_dim = args.input_dim
    if args.num_classes is not None:
        config.graph.num_classes = args.num_classes
    if args.train_fraction is not None:
        config.graph.train_fraction = args.train_fraction
    if args.seed is not None:
        config.graph.seed = args.seed
    if args.hidden_dim is not None:
        config.model.hidden_dim = args.hidden_dim
    if args.num_layers is not None:
        config.model.num_layers = args.num_layers
    if args.dropout is not None:
        config.model.dropout = args.dropout
    if args.edge_chunk_size is not None:
        config.model.edge_chunk_size = args.edge_chunk_size
    if args.ffn_multiplier is not None:
        config.model.ffn_multiplier = args.ffn_multiplier
    if args.lr is not None:
        config.optimizer.lr = args.lr
    if args.weight_decay is not None:
        config.optimizer.weight_decay = args.weight_decay
    if args.optimizer is not None:
        config.optimizer.name = args.optimizer
    return config


def validate_config(config: BenchmarkConfig) -> None:
    graph = config.graph
    model = config.model
    run = config.run
    sizing = config.sizing

    if graph.num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if graph.num_relations <= 0:
        raise ValueError("num_relations must be positive")
    if graph.edges_per_relation <= 0:
        raise ValueError("edges_per_relation must be positive")
    if graph.input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if graph.num_classes <= 1:
        raise ValueError("num_classes must be greater than 1")
    if not 0.0 < graph.train_fraction <= 1.0:
        raise ValueError("train_fraction must be in the interval (0, 1]")
    if model.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if model.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if not 0.0 <= model.dropout < 1.0:
        raise ValueError("dropout must be in the interval [0, 1)")
    if model.edge_chunk_size <= 0:
        raise ValueError("edge_chunk_size must be positive")
    if model.ffn_multiplier <= 0:
        raise ValueError("ffn_multiplier must be positive")
    if run.epochs <= 0:
        raise ValueError("epochs must be positive")
    if run.min_duration_sec < 0.0:
        raise ValueError("min_duration_sec cannot be negative")
    if run.warmup_epochs < 0:
        raise ValueError("warmup_epochs cannot be negative")
    if run.log_every <= 0:
        raise ValueError("log_every must be positive")
    if run.precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be one of fp32, fp16, bf16")
    if sizing.target_memory_gb is not None and sizing.target_memory_gb <= 0.0:
        raise ValueError("target_memory_gb must be positive")
    if sizing.target_memory_fraction is not None and not 0.0 < sizing.target_memory_fraction <= 1.0:
        raise ValueError("target_memory_fraction must be in the interval (0, 1]")
    if not 0.0 < sizing.tolerance < 1.0:
        raise ValueError("sizing tolerance must be in the interval (0, 1)")
    if sizing.max_probes <= 0:
        raise ValueError("sizing max_probes must be positive")


def resolve_device(requested_device: str) -> tuple[Any, str]:
    ensure_mlx_available()
    gpu_count = int(mx.device_count(mx.gpu))

    if requested_device == "auto":
        return (mx.gpu, "gpu") if gpu_count > 0 else (mx.cpu, "cpu")
    if requested_device == "gpu" and gpu_count <= 0:
        raise RuntimeError("MLX GPU requested but no GPU device is available")
    return (mx.gpu, "gpu") if requested_device == "gpu" else (mx.cpu, "cpu")


def physical_memory_gb() -> float | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return (page_size * page_count) / (1024**3)


def mlx_version() -> str | None:
    try:
        return importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_hardware_info(device_name: str) -> dict[str, Any]:
    host_memory_gb = physical_memory_gb()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "mlx_version": mlx_version(),
        "backend": device_name,
        "device_type": device_name,
        "device_name": f"{platform.processor() or 'Apple Silicon'} {device_name.upper()}",
        "host_memory_gb": host_memory_gb,
        "total_device_memory_gb": host_memory_gb,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def configure_runtime(device: Any) -> None:
    ensure_mlx_available()
    mx.set_default_device(device)
    mx.set_default_stream(mx.default_stream(device))


def synchronize_device() -> None:
    ensure_mlx_available()
    mx.synchronize()


def reset_peak_memory_stats() -> None:
    ensure_mlx_available()
    mx.reset_peak_memory()


def read_memory_stats() -> dict[str, float | None]:
    ensure_mlx_available()
    return {
        "allocated_gb": mx.get_active_memory() / (1024**3),
        "reserved_gb": mx.get_cache_memory() / (1024**3),
        "peak_allocated_gb": mx.get_peak_memory() / (1024**3),
        "peak_reserved_gb": None,
        "driver_allocated_gb": None,
    }


def available_memory_gb(hardware: dict[str, Any]) -> float | None:
    total_device_memory_gb = hardware.get("total_device_memory_gb")
    if total_device_memory_gb is not None:
        return float(total_device_memory_gb)
    return hardware.get("host_memory_gb")


def is_oom_error(error: BaseException) -> bool:
    if isinstance(error, MemoryError):
        return True
    if "outofmemory" in type(error).__name__.lower():
        return True
    message = str(error).lower()
    return "out of memory" in message or "not enough memory" in message


def cleanup_device_memory() -> None:
    gc.collect()
    if mx is not None and hasattr(mx, "clear_cache"):
        mx.clear_cache()


def peak_memory_value(memory_stats: dict[str, float | None]) -> float | None:
    for key in ("peak_allocated_gb", "allocated_gb", "reserved_gb"):
        value = memory_stats.get(key)
        if value is not None:
            return value
    return None


def precision_to_dtype(precision: str) -> Any:
    ensure_mlx_available()
    if precision == "fp32":
        return mx.float32
    if precision == "fp16":
        return mx.float16
    if hasattr(mx, "bfloat16"):
        return mx.bfloat16
    raise ValueError("MLX bfloat16 is not available in this environment")


def build_epoch_estimates(config: BenchmarkConfig) -> dict[str, float]:
    storage_bytes = benchmark_utils.precision_bytes(config.run.precision)
    return benchmark_utils.estimate_epoch_telemetry(
        config.graph,
        config.model,
        config.run.precision,
        feature_storage_bytes=storage_bytes,
        parameter_storage_bytes=storage_bytes,
        index_dtype_bytes=4,
        label_dtype_bytes=4,
    )


def build_synthetic_graph(config: GraphConfig, dtype: Any) -> SyntheticGraph:
    ensure_mlx_available()
    print("Preparing synthetic graph for MLX...", flush=True)
    rng = np.random.default_rng(config.seed)

    features = rng.standard_normal((config.num_nodes, config.input_dim)).astype(np.float32)
    teacher = rng.standard_normal((config.input_dim, config.num_classes)).astype(np.float32)
    labels = (features @ teacher).argmax(axis=1).astype(np.int32)
    train_count = max(1, int(config.num_nodes * config.train_fraction))
    train_index = rng.permutation(config.num_nodes)[:train_count].astype(np.int32)

    features_array = mx.array(features)
    if dtype != mx.float32:
        features_array = features_array.astype(dtype)

    labels_array = mx.array(labels)
    train_index_array = mx.array(train_index)
    train_labels = labels_array[train_index_array]

    relations: list[tuple[Any, Any, Any]] = []
    for relation_id in range(config.num_relations):
        print(
            f"  relation {relation_id + 1}/{config.num_relations}: "
            f"{config.edges_per_relation:,} edges",
            flush=True,
        )
        source = rng.integers(0, config.num_nodes, size=config.edges_per_relation, dtype=np.int32)
        destination = rng.integers(0, config.num_nodes, size=config.edges_per_relation, dtype=np.int32)
        degree = np.bincount(destination, minlength=config.num_nodes).astype(np.float32)
        normalization = (1.0 / np.maximum(degree[destination], 1.0)).astype(np.float32)

        source_array = mx.array(source)
        destination_array = mx.array(destination)
        normalization_array = mx.array(normalization)
        if dtype != mx.float32:
            normalization_array = normalization_array.astype(dtype)
        relations.append((source_array, destination_array, normalization_array))

    return SyntheticGraph(
        features=features_array,
        labels=labels_array,
        train_index=train_index_array,
        train_labels=train_labels,
        relations=relations,
        total_edges=config.num_relations * config.edges_per_relation,
    )


def glorot_uniform(shape: tuple[int, ...], dtype: Any) -> Any:
    ensure_mlx_available()
    if len(shape) < 2:
        raise ValueError("glorot_uniform expects at least two dimensions")
    fan_in = shape[-2]
    fan_out = shape[-1]
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    values = (mx.random.uniform(shape=shape) * (2.0 * limit)) - limit
    return values.astype(dtype) if dtype != mx.float32 else values


def aggregate_by_destination(aggregated: Any, destination: Any, messages: Any) -> Any:
    edge_count = int(destination.shape[0])
    if edge_count == 0:
        return aggregated

    order = mx.argsort(destination)
    sorted_destination = destination[order]
    sorted_messages = messages[order]

    if edge_count == 1:
        unique_destination = sorted_destination
        segment_sums = sorted_messages
    else:
        changes = sorted_destination[1:] != sorted_destination[:-1]
        first_true = mx.array(np.array([True]))
        boundaries = mx.concatenate([first_true, changes], axis=0)
        unique_destination = sorted_destination[boundaries]

        positions = mx.arange(edge_count - 1, dtype=destination.dtype)
        last_position = mx.array(np.array([edge_count - 1], dtype=np.int32))
        end_positions = mx.concatenate([positions[changes], last_position], axis=0)
        first_position = mx.array(np.array([0], dtype=np.int32))
        start_positions = mx.concatenate([first_position, end_positions[:-1] + 1], axis=0)

        prefix = mx.cumsum(sorted_messages, axis=0)
        prefix_prev = mx.concatenate(
            [mx.zeros((1, messages.shape[1]), dtype=messages.dtype), prefix[:-1]],
            axis=0,
        )
        segment_sums = prefix[end_positions] - prefix_prev[start_positions]

    aggregated[unique_destination] = aggregated[unique_destination] + segment_sums
    return aggregated


def probe_graph_memory(
    config: BenchmarkConfig,
    candidate_graph_config: GraphConfig,
    scale: float,
    probe_index: int,
) -> benchmark_utils.AutosizeProbe:
    graph = None
    model = None
    optimizer = None
    loss_and_grad_fn = None
    loss_value = None
    accuracy_value = None
    grads = None

    try:
        dtype = precision_to_dtype(config.run.precision)
        graph = build_synthetic_graph(candidate_graph_config, dtype)
        model = RGCNModel(candidate_graph_config, config.model, dtype)
        optimizer = build_optimizer(config.optimizer)
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        mx.eval(model.parameters())

        model.train()
        reset_peak_memory_stats()
        synchronize_device()
        probe_started = time.perf_counter()
        (loss_value, accuracy_value), grads = loss_and_grad_fn(model, graph)
        optimizer.update(model, grads)
        mx.eval(loss_value, accuracy_value, model.parameters(), optimizer.state)
        synchronize_device()
        epoch_seconds = time.perf_counter() - probe_started
        memory_stats = read_memory_stats()

        return benchmark_utils.AutosizeProbe(
            probe=probe_index,
            scale=scale,
            num_nodes=candidate_graph_config.num_nodes,
            edges_per_relation=candidate_graph_config.edges_per_relation,
            measured_memory_gb=benchmark_utils.effective_memory_usage_gb(memory_stats),
            epoch_seconds=epoch_seconds,
            oom=False,
        )
    except Exception as error:
        if not is_oom_error(error):
            raise
        return benchmark_utils.AutosizeProbe(
            probe=probe_index,
            scale=scale,
            num_nodes=candidate_graph_config.num_nodes,
            edges_per_relation=candidate_graph_config.edges_per_relation,
            measured_memory_gb=None,
            epoch_seconds=None,
            oom=True,
            note=str(error).splitlines()[0][:200],
        )
    finally:
        del graph, model, optimizer, loss_and_grad_fn, loss_value, accuracy_value, grads
        cleanup_device_memory()


def maybe_auto_size_config(
    config: BenchmarkConfig,
    hardware: dict[str, Any],
) -> tuple[BenchmarkConfig, dict[str, Any]]:
    if not config.sizing.enabled:
        return config, {
            "enabled": False,
            "available_memory_gb": available_memory_gb(hardware),
            "target_memory_gb": config.sizing.target_memory_gb,
            "probe_history": [],
        }

    print(
        "Autosizing graph toward the requested unified-memory budget...",
        flush=True,
    )

    def probe_fn(
        candidate_graph_config: GraphConfig,
        scale: float,
        probe_index: int,
    ) -> benchmark_utils.AutosizeProbe:
        probe = probe_graph_memory(
            config,
            candidate_graph_config,
            scale,
            probe_index,
        )
        memory_fragment = (
            f"{probe.measured_memory_gb:.2f}GB"
            if probe.measured_memory_gb is not None
            else "n/a"
        )
        epoch_fragment = (
            f"{probe.epoch_seconds:.2f}s" if probe.epoch_seconds is not None else "n/a"
        )
        note_fragment = f" note={probe.note}" if probe.note else ""
        status = "oom" if probe.oom else "ok"
        print(
            f"autosize probe {probe.probe}/{config.sizing.max_probes}: "
            f"scale={probe.scale:.3f} "
            f"nodes={probe.num_nodes:,} "
            f"edges/rel={probe.edges_per_relation:,} "
            f"mem={memory_fragment} "
            f"epoch_s={epoch_fragment} "
            f"status={status}{note_fragment}",
            flush=True,
        )
        return probe

    graph_config, autosize_summary = benchmark_utils.auto_size_graph(
        config.graph,
        config.sizing,
        available_memory_gb(hardware),
        probe_fn,
    )
    measured_memory = autosize_summary.get("final_measured_memory_gb")
    measured_fragment = (
        f"{measured_memory:.2f}GB" if measured_memory is not None else "n/a"
    )
    print(
        f"Autosize selected scale={autosize_summary['final_scale']:.3f} "
        f"nodes={graph_config.num_nodes:,} "
        f"edges/rel={graph_config.edges_per_relation:,} "
        f"target={autosize_summary['target_memory_gb']:.2f}GB "
        f"measured={measured_fragment}",
        flush=True,
    )
    return replace(config, graph=graph_config), autosize_summary


class RGCNBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_relations: int,
        dropout: float,
        edge_chunk_size: int,
        ffn_multiplier: int,
        dtype: Any,
    ) -> None:
        super().__init__()
        self.edge_chunk_size = edge_chunk_size
        self.self_loop = nn.Linear(input_dims=hidden_dim, output_dims=hidden_dim, bias=False)
        self.relation_weights = glorot_uniform((num_relations, hidden_dim, hidden_dim), dtype)
        self.output_norm = nn.LayerNorm(dims=hidden_dim)
        self.ffn_norm = nn.LayerNorm(dims=hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        inner_dim = hidden_dim * ffn_multiplier
        self.ffn_in = nn.Linear(input_dims=hidden_dim, output_dims=inner_dim, bias=False)
        self.ffn_out = nn.Linear(input_dims=inner_dim, output_dims=hidden_dim, bias=False)

    def __call__(self, x: Any, relations: list[tuple[Any, Any, Any]]) -> Any:
        aggregated = mx.zeros_like(x)

        for relation_index, (source, destination, normalization) in enumerate(relations):
            weight = self.relation_weights[relation_index]
            edge_count = int(source.shape[0])

            for start in range(0, edge_count, self.edge_chunk_size):
                stop = min(start + self.edge_chunk_size, edge_count)
                source_states = x[source[start:stop]]
                messages = mx.matmul(source_states, weight)
                messages = messages * mx.expand_dims(normalization[start:stop], axis=1)
                aggregated = aggregate_by_destination(aggregated, destination[start:stop], messages)

        x = self.output_norm(aggregated + self.self_loop(x) + x)
        x = nn.gelu(x)
        x = self.dropout(x)

        residual = x
        x = self.ffn_in(self.ffn_norm(x))
        x = nn.gelu(x)
        x = self.ffn_out(x)
        x = self.dropout(x)
        return residual + x


class RGCNModel(nn.Module):
    def __init__(self, graph_config: GraphConfig, model_config: ModelConfig, dtype: Any) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(
                input_dims=graph_config.input_dim,
                output_dims=model_config.hidden_dim,
                bias=False,
            ),
            nn.LayerNorm(dims=model_config.hidden_dim),
            nn.GELU(),
            nn.Dropout(p=model_config.dropout),
        )
        self.layers = [
            RGCNBlock(
                hidden_dim=model_config.hidden_dim,
                num_relations=graph_config.num_relations,
                dropout=model_config.dropout,
                edge_chunk_size=model_config.edge_chunk_size,
                ffn_multiplier=model_config.ffn_multiplier,
                dtype=dtype,
            )
            for _ in range(model_config.num_layers)
        ]
        self.classifier_norm = nn.LayerNorm(dims=model_config.hidden_dim)
        self.classifier = nn.Linear(
            input_dims=model_config.hidden_dim,
            output_dims=graph_config.num_classes,
            bias=True,
        )
        self.set_dtype(dtype)

    def __call__(self, features: Any, relations: list[tuple[Any, Any, Any]]) -> Any:
        x = self.input_projection(features)
        for layer in self.layers:
            x = layer(x, relations)
        return self.classifier(self.classifier_norm(x))


def build_optimizer(config: OptimizerConfig):
    if config.name == "adamw":
        return optim.AdamW(learning_rate=config.lr, weight_decay=config.weight_decay)
    if config.name == "sgd":
        return optim.SGD(
            learning_rate=config.lr,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.name}")


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(value: str) -> str:
    collapsed = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return collapsed.strip("-") or "run"


def create_run_dir(config: BenchmarkConfig, backend: str) -> Path:
    root = Path(config.run.results_dir)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"{timestamp}_{slugify(config.name)}_{backend}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def steady_state_records(history: list[dict[str, Any]], warmup_epochs: int) -> list[dict[str, Any]]:
    steady = [record for record in history if record["epoch"] > warmup_epochs]
    return steady if steady else history


def summarize_history(
    history: list[dict[str, Any]],
    config: BenchmarkConfig,
    graph: SyntheticGraph,
) -> dict[str, Any]:
    steady = steady_state_records(history, config.run.warmup_epochs)
    epoch_seconds = [record["epoch_seconds"] for record in steady]
    message_rates = [record["message_edge_updates_per_sec"] for record in steady]
    accuracy_values = [record["train_accuracy"] for record in steady]
    peak_values = [record["peak_memory_gb"] for record in history if record["peak_memory_gb"] is not None]

    return {
        "epochs_completed": history[-1]["epoch"],
        "warmup_epochs_excluded": config.run.warmup_epochs,
        "training_duration_sec": history[-1]["elapsed_seconds"],
        "setup_duration_sec": history[0]["setup_seconds"],
        "steady_state_epoch_sec_median": statistics.median(epoch_seconds),
        "steady_state_epoch_sec_mean": statistics.fmean(epoch_seconds),
        "steady_state_message_edge_updates_per_sec_median": statistics.median(message_rates),
        "steady_state_message_edge_updates_per_sec_mean": statistics.fmean(message_rates),
        "steady_state_train_accuracy_mean": statistics.fmean(accuracy_values),
        "final_loss": history[-1]["loss"],
        "peak_memory_gb": max(peak_values) if peak_values else None,
        "train_nodes": int(graph.train_index.shape[0]),
    }


def write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return

    fieldnames = list(history[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def format_epoch_line(record: dict[str, Any]) -> str:
    peak_memory = record["peak_memory_gb"]
    peak_fragment = (
        f"peak_mem={peak_memory:.2f}GB" if peak_memory is not None else "peak_mem=n/a"
    )
    return (
        f"epoch={record['epoch']:04d} "
        f"loss={record['loss']:.4f} "
        f"acc={record['train_accuracy']:.4f} "
        f"epoch_s={record['epoch_seconds']:.2f} "
        f"edge_updates/s={record['message_edge_updates_per_sec'] / 1e6:.2f}M "
        f"elapsed_s={record['elapsed_seconds']:.2f} "
        f"{peak_fragment}"
    )


def run_benchmark(config: BenchmarkConfig, device: Any, device_name: str) -> Path:
    ensure_mlx_available()
    started_at = benchmark_utils.iso_utc_now()
    setup_start = time.perf_counter()

    seed_everything(config.graph.seed)
    configure_runtime(device)
    hardware = collect_hardware_info(device_name)
    config, autosize_summary = maybe_auto_size_config(config, hardware)
    dtype = precision_to_dtype(config.run.precision)
    graph = build_synthetic_graph(config.graph, dtype)
    model = RGCNModel(config.graph, config.model, dtype)
    optimizer = build_optimizer(config.optimizer)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    epoch_estimates = build_epoch_estimates(config)

    mx.eval(model.parameters())
    synchronize_device()
    setup_seconds = time.perf_counter() - setup_start
    run_dir = benchmark_utils.create_run_dir(
        config.run.results_dir,
        config.name,
        f"mlx-{device_name}",
    )

    command = " ".join(shlex.quote(arg) for arg in sys.argv)
    message_edges_per_epoch = graph.total_edges * config.model.num_layers

    print(f"Run directory: {run_dir}", flush=True)
    print("Framework: mlx", flush=True)
    print(f"Backend: {device_name}", flush=True)
    print(f"Device: {hardware['device_name']}", flush=True)
    print(
        "Workload: "
        f"nodes={config.graph.num_nodes:,}, "
        f"relations={config.graph.num_relations}, "
        f"edges={graph.total_edges:,}, "
        f"hidden_dim={config.model.hidden_dim}, "
        f"layers={config.model.num_layers}, "
        f"precision={config.run.precision}",
        flush=True,
    )
    print(
        f"Training will continue until epoch>={config.run.epochs} and "
        f"elapsed>={config.run.min_duration_sec:.0f}s",
        flush=True,
    )

    history: list[dict[str, Any]] = []
    training_start = time.perf_counter()
    epoch = 0

    while True:
        epoch += 1
        model.train()
        reset_peak_memory_stats()
        synchronize_device()
        epoch_start = time.perf_counter()

        (loss_value, accuracy_value), grads = loss_and_grad_fn(model, graph)
        optimizer.update(model, grads)
        mx.eval(loss_value, accuracy_value, model.parameters(), optimizer.state)
        synchronize_device()

        epoch_seconds = time.perf_counter() - epoch_start
        elapsed_seconds = time.perf_counter() - training_start
        memory_stats = read_memory_stats()

        record = {
            "epoch": epoch,
            "loss": float(loss_value.item()),
            "train_accuracy": float(accuracy_value.item()),
            "epoch_seconds": epoch_seconds,
            "elapsed_seconds": elapsed_seconds,
            "message_edge_updates_per_sec": message_edges_per_epoch / epoch_seconds,
            "allocated_gb": memory_stats["allocated_gb"],
            "reserved_gb": memory_stats["reserved_gb"],
            "peak_allocated_gb": memory_stats["peak_allocated_gb"],
            "peak_reserved_gb": memory_stats["peak_reserved_gb"],
            "driver_allocated_gb": memory_stats["driver_allocated_gb"],
            "effective_memory_gb": benchmark_utils.effective_memory_usage_gb(
                memory_stats
            ),
            "peak_memory_gb": peak_memory_value(memory_stats),
            "setup_seconds": setup_seconds,
        }
        benchmark_utils.attach_epoch_telemetry(record, epoch_estimates)
        history.append(record)

        if epoch == 1 or epoch % config.run.log_every == 0:
            print(benchmark_utils.format_epoch_line(record), flush=True)

        if epoch >= config.run.epochs and elapsed_seconds >= config.run.min_duration_sec:
            break

    finished_at = benchmark_utils.iso_utc_now()
    summary = {
        "run": {
            "name": config.name,
            "description": config.description,
            "framework": "mlx",
            "backend": device_name,
            "device": device_name,
            "precision": config.run.precision,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "command": command,
            "results_dir": str(run_dir),
        },
        "hardware": hardware,
        "workload": {
            "num_nodes": config.graph.num_nodes,
            "num_relations": config.graph.num_relations,
            "edges_per_relation": config.graph.edges_per_relation,
            "total_edges": graph.total_edges,
            "input_dim": config.graph.input_dim,
            "num_classes": config.graph.num_classes,
            "hidden_dim": config.model.hidden_dim,
            "num_layers": config.model.num_layers,
            "edge_chunk_size": config.model.edge_chunk_size,
            "ffn_multiplier": config.model.ffn_multiplier,
            "message_edges_per_epoch": message_edges_per_epoch,
            "graph_scale": autosize_summary.get("final_scale", 1.0),
        },
        "metrics": benchmark_utils.summarize_history(
            history,
            config.run.warmup_epochs,
            int(graph.train_index.shape[0]),
        ),
        "autosize": autosize_summary,
        "config": asdict(config),
    }

    benchmark_utils.write_history_csv(run_dir / "history.csv", history)
    benchmark_utils.write_summary_json(run_dir / "summary.json", summary)
    print(f"Summary written to {run_dir / 'summary.json'}", flush=True)
    return run_dir


def loss_fn(model: Any, graph: SyntheticGraph) -> tuple[Any, Any]:
    logits = model(graph.features, graph.relations)
    train_logits = logits[graph.train_index]
    loss = mx.mean(nn.losses.cross_entropy(train_logits, graph.train_labels))
    accuracy = mx.mean(mx.argmax(train_logits, axis=1) == graph.train_labels)
    return loss, accuracy


def main() -> int:
    ensure_mlx_available()
    args = parse_args()
    config = apply_cli_overrides(load_benchmark_config(args.config), args)
    validate_config(config)
    device, device_name = resolve_device(args.device)
    run_benchmark(config, device, device_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
