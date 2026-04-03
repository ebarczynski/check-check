from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import re
import resource
import shlex
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from rgcn_benchmark import benchmark_utils
from torch import Tensor, nn


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
    tf32: bool = True


@dataclass
class BenchmarkConfig:
    name: str = "rgcn_portable_long"
    description: str = "Portable PyTorch RGCN benchmark"
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    run: RunConfig = field(default_factory=RunConfig)
    sizing: benchmark_utils.SizingConfig = field(
        default_factory=benchmark_utils.SizingConfig
    )


@dataclass
class SyntheticGraph:
    features: Tensor
    labels: Tensor
    train_index: Tensor
    train_labels: Tensor
    relations: list[tuple[Tensor, Tensor, Tensor]]
    total_edges: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a backend-portable PyTorch RGCN workload."
    )
    parser.add_argument("--config", type=Path, help="Path to a JSON config preset.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "mps", "cpu"),
        help="Target device. ROCm uses 'cuda' in PyTorch.",
    )
    parser.add_argument("--name", help="Override the run name.")
    parser.add_argument("--description", help="Override the run description.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Override the results directory.",
    )
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
        help="Probe memory headroom and scale the graph toward a target memory budget before the full run.",
    )
    parser.add_argument(
        "--target-memory-gb",
        type=float,
        help="Absolute VRAM or unified-memory budget to target during autosizing.",
    )
    parser.add_argument(
        "--target-memory-fraction",
        type=float,
        help="Fraction of available memory to target during autosizing.",
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
    parser.add_argument(
        "--num-relations", type=int, help="Override number of edge relations."
    )
    parser.add_argument(
        "--edges-per-relation",
        type=int,
        help="Override edges per relation.",
    )
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
    parser.add_argument(
        "--dropout",
        type=float,
        help="Override dropout probability.",
    )
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
    parser.add_argument(
        "--weight-decay", type=float, help="Override optimizer weight decay."
    )
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "sgd"),
        help="Override optimizer type.",
    )
    parser.add_argument(
        "--disable-tf32",
        action="store_true",
        help="Disable TF32 acceleration on CUDA backends.",
    )
    return parser.parse_args()


def load_benchmark_config(path: Path | None) -> BenchmarkConfig:
    config = BenchmarkConfig()
    if path is None:
        return config

    payload = json.loads(path.read_text(encoding="utf-8"))
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
    if args.disable_tf32:
        config.run.tf32 = False
    return config


def validate_config(config: BenchmarkConfig, device: torch.device) -> None:
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
    if device.type == "cpu" and run.precision != "fp32":
        raise ValueError("CPU runs currently support only fp32 in this benchmark")
    if device.type == "mps" and run.precision == "bf16":
        raise ValueError("MPS runs should use fp32 or fp16, not bf16")


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA or ROCm device requested but not available")
    if requested_device == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS device requested but not available")
    return torch.device(requested_device)


def backend_label(device: torch.device) -> str:
    if device.type == "cuda":
        return "rocm" if torch.version.hip is not None else "cuda"
    return device.type


def physical_memory_gb() -> float | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return (page_size * page_count) / (1024**3)


def collect_hardware_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "backend": backend_label(device),
        "device_type": device.type,
        "host_memory_gb": physical_memory_gb(),
    }

    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        info.update(
            {
                "device_name": torch.cuda.get_device_name(device_index),
                "total_device_memory_gb": props.total_memory / (1024**3),
                "multiprocessor_count": props.multi_processor_count,
                "cuda_runtime": torch.version.cuda,
                "hip_runtime": torch.version.hip,
            }
        )
    elif device.type == "mps":
        recommended_max_memory_gb = None
        if hasattr(torch.mps, "recommended_max_memory"):
            recommended_max_memory_gb = (
                torch.mps.recommended_max_memory() / (1024**3)
            )
        info.update(
            {
                "device_name": platform.processor() or "Apple Silicon",
                "mps_available": torch.backends.mps.is_available(),
                "mps_built": torch.backends.mps.is_built(),
                "recommended_max_memory_gb": recommended_max_memory_gb,
                "total_device_memory_gb": recommended_max_memory_gb,
            }
        )
    else:
        info["device_name"] = platform.processor() or "CPU"
        info["total_device_memory_gb"] = info["host_memory_gb"]

    return info


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(config: BenchmarkConfig, device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = config.run.tf32
        torch.backends.cudnn.allow_tf32 = config.run.tf32


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def process_resident_memory_gb() -> float | None:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    resident_kb = float(line.split()[1])
                    return resident_kb / (1024**2)
        except OSError:
            return None
    return None


def process_peak_memory_gb() -> float | None:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage <= 0:
        return None
    if sys.platform == "darwin":
        return usage / (1024**3)
    return usage / (1024**2)


def read_memory_stats(device: torch.device) -> dict[str, float | None]:
    stats: dict[str, float | None] = {
        "allocated_gb": None,
        "reserved_gb": None,
        "peak_allocated_gb": None,
        "peak_reserved_gb": None,
        "driver_allocated_gb": None,
    }

    if device.type == "cuda":
        stats["allocated_gb"] = torch.cuda.memory_allocated(device) / (1024**3)
        stats["reserved_gb"] = torch.cuda.memory_reserved(device) / (1024**3)
        stats["peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / (1024**3)
        stats["peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / (1024**3)
    elif device.type == "mps":
        if hasattr(torch.mps, "current_allocated_memory"):
            stats["allocated_gb"] = torch.mps.current_allocated_memory() / (1024**3)
        if hasattr(torch.mps, "driver_allocated_memory"):
            stats["driver_allocated_gb"] = torch.mps.driver_allocated_memory() / (1024**3)
    else:
        stats["allocated_gb"] = process_resident_memory_gb()
        stats["reserved_gb"] = stats["allocated_gb"]
        stats["peak_allocated_gb"] = process_peak_memory_gb()

    return stats


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


def cleanup_device_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def build_epoch_estimates(config: BenchmarkConfig) -> dict[str, float]:
    return benchmark_utils.estimate_epoch_telemetry(
        config.graph,
        config.model,
        config.run.precision,
        feature_storage_bytes=4,
        parameter_storage_bytes=4,
        index_dtype_bytes=8,
        label_dtype_bytes=8,
    )


def probe_graph_memory(
    config: BenchmarkConfig,
    device: torch.device,
    candidate_graph_config: GraphConfig,
    scale: float,
    probe_index: int,
) -> benchmark_utils.AutosizeProbe:
    graph = None
    model = None
    optimizer = None
    scaler = None
    logits = None
    train_logits = None
    loss = None

    try:
        graph = build_synthetic_graph(candidate_graph_config, device)
        model = RGCNModel(candidate_graph_config, config.model).to(device)
        optimizer = build_optimizer(config.optimizer, model)
        scaler = make_grad_scaler(device, config.run.precision)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        reset_peak_memory_stats(device)
        synchronize_device(device)
        probe_started = time.perf_counter()

        with autocast_context(device, config.run.precision):
            logits = model(graph.features, graph.relations)
            train_logits = logits.index_select(0, graph.train_index)
            loss = F.cross_entropy(train_logits, graph.train_labels)

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        synchronize_device(device)
        epoch_seconds = time.perf_counter() - probe_started
        memory_stats = read_memory_stats(device)
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
        del graph, model, optimizer, scaler, logits, train_logits, loss
        cleanup_device_memory(device)


def maybe_auto_size_config(
    config: BenchmarkConfig,
    device: torch.device,
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
        "Autosizing graph toward the requested memory budget...",
        flush=True,
    )

    def probe_fn(
        candidate_graph_config: GraphConfig,
        scale: float,
        probe_index: int,
    ) -> benchmark_utils.AutosizeProbe:
        probe = probe_graph_memory(
            config,
            device,
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


def peak_memory_value(memory_stats: dict[str, float | None]) -> float | None:
    for key in ("peak_allocated_gb", "allocated_gb", "driver_allocated_gb"):
        value = memory_stats.get(key)
        if value is not None:
            return value
    return None


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_grad_scaler(device: torch.device, precision: str):
    if device.type != "cuda" or precision != "fp16":
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    return torch.cuda.amp.GradScaler()


def build_synthetic_graph(config: GraphConfig, device: torch.device) -> SyntheticGraph:
    print("Preparing synthetic graph...", flush=True)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    features = torch.randn(
        (config.num_nodes, config.input_dim),
        generator=generator,
        dtype=torch.float32,
    )
    teacher = torch.randn(
        (config.input_dim, config.num_classes),
        generator=generator,
        dtype=torch.float32,
    )
    labels = (features @ teacher).argmax(dim=1).to(torch.long)

    train_count = max(1, int(config.num_nodes * config.train_fraction))
    train_index = torch.randperm(config.num_nodes, generator=generator)[:train_count]
    edge_weight = torch.ones(config.edges_per_relation, dtype=torch.float32)

    features = features.to(device)
    labels = labels.to(device)
    train_index = train_index.to(device)
    train_labels = labels.index_select(0, train_index)

    relations: list[tuple[Tensor, Tensor, Tensor]] = []
    for relation_id in range(config.num_relations):
        print(
            f"  relation {relation_id + 1}/{config.num_relations}: "
            f"{config.edges_per_relation:,} edges",
            flush=True,
        )
        source = torch.randint(
            config.num_nodes,
            (config.edges_per_relation,),
            generator=generator,
            dtype=torch.int64,
        )
        destination = torch.randint(
            config.num_nodes,
            (config.edges_per_relation,),
            generator=generator,
            dtype=torch.int64,
        )
        degree = torch.zeros(config.num_nodes, dtype=torch.float32)
        degree.index_add_(0, destination, edge_weight)
        normalization = degree.index_select(0, destination).clamp_min_(1.0).reciprocal()
        relations.append(
            (
                source.to(device),
                destination.to(device),
                normalization.to(device),
            )
        )

    return SyntheticGraph(
        features=features,
        labels=labels,
        train_index=train_index,
        train_labels=train_labels,
        relations=relations,
        total_edges=config.num_relations * config.edges_per_relation,
    )


class RGCNBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_relations: int,
        dropout: float,
        edge_chunk_size: int,
        ffn_multiplier: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.edge_chunk_size = edge_chunk_size

        self.self_loop = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation_weights = nn.Parameter(
            torch.empty(num_relations, hidden_dim, hidden_dim)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        inner_dim = hidden_dim * ffn_multiplier
        self.ffn_in = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.ffn_out = nn.Linear(inner_dim, hidden_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.self_loop.weight)
        nn.init.xavier_uniform_(self.relation_weights)
        nn.init.xavier_uniform_(self.ffn_in.weight)
        nn.init.xavier_uniform_(self.ffn_out.weight)

    def forward(self, x: Tensor, relations: list[tuple[Tensor, Tensor, Tensor]]) -> Tensor:
        aggregated = x.new_zeros(x.shape)

        for relation_index, (source, destination, normalization) in enumerate(relations):
            weight = self.relation_weights[relation_index]
            edge_count = int(source.numel())

            for start in range(0, edge_count, self.edge_chunk_size):
                stop = min(start + self.edge_chunk_size, edge_count)
                source_states = x.index_select(0, source[start:stop])
                messages = source_states.matmul(weight)
                normalization_slice = normalization[start:stop].unsqueeze(1).to(
                    dtype=messages.dtype
                )
                messages = messages * normalization_slice
                aggregated.index_add_(0, destination[start:stop], messages)

        x = self.output_norm(aggregated + self.self_loop(x) + x)
        x = F.gelu(x)
        x = self.dropout(x)

        residual = x
        x = self.ffn_in(self.ffn_norm(x))
        x = F.gelu(x)
        x = self.ffn_out(x)
        x = self.dropout(x)
        return residual + x


class RGCNModel(nn.Module):
    def __init__(self, graph_config: GraphConfig, model_config: ModelConfig) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(graph_config.input_dim, model_config.hidden_dim, bias=False),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
        )
        self.layers = nn.ModuleList(
            [
                RGCNBlock(
                    hidden_dim=model_config.hidden_dim,
                    num_relations=graph_config.num_relations,
                    dropout=model_config.dropout,
                    edge_chunk_size=model_config.edge_chunk_size,
                    ffn_multiplier=model_config.ffn_multiplier,
                )
                for _ in range(model_config.num_layers)
            ]
        )
        self.classifier_norm = nn.LayerNorm(model_config.hidden_dim)
        self.classifier = nn.Linear(model_config.hidden_dim, graph_config.num_classes)

    def forward(self, features: Tensor, relations: list[tuple[Tensor, Tensor, Tensor]]) -> Tensor:
        x = self.input_projection(features)
        for layer in self.layers:
            x = layer(x, relations)
        return self.classifier(self.classifier_norm(x))


def build_optimizer(config: OptimizerConfig, model: nn.Module):
    if config.name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    if config.name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=0.9,
        )
    raise ValueError(f"Unsupported optimizer: {config.name}")


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify(value: str) -> str:
    collapsed = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return collapsed.strip("-") or "run"


def create_run_dir(config: BenchmarkConfig, device: torch.device) -> Path:
    root = Path(config.run.results_dir)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"{timestamp}_{slugify(config.name)}_{backend_label(device)}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def steady_state_records(
    history: list[dict[str, Any]], warmup_epochs: int
) -> list[dict[str, Any]]:
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
        "steady_state_message_edge_updates_per_sec_median": statistics.median(
            message_rates
        ),
        "steady_state_message_edge_updates_per_sec_mean": statistics.fmean(
            message_rates
        ),
        "steady_state_train_accuracy_mean": statistics.fmean(accuracy_values),
        "final_loss": history[-1]["loss"],
        "peak_memory_gb": max(peak_values) if peak_values else None,
        "train_nodes": int(graph.train_index.numel()),
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


def run_benchmark(config: BenchmarkConfig, device: torch.device) -> Path:
    started_at = benchmark_utils.iso_utc_now()
    setup_start = time.perf_counter()

    seed_everything(config.graph.seed)
    configure_runtime(config, device)
    hardware = collect_hardware_info(device)
    config, autosize_summary = maybe_auto_size_config(config, device, hardware)
    graph = build_synthetic_graph(config.graph, device)
    model = RGCNModel(config.graph, config.model).to(device)
    optimizer = build_optimizer(config.optimizer, model)
    scaler = make_grad_scaler(device, config.run.precision)
    epoch_estimates = build_epoch_estimates(config)

    synchronize_device(device)
    setup_seconds = time.perf_counter() - setup_start
    run_dir = benchmark_utils.create_run_dir(
        config.run.results_dir,
        config.name,
        backend_label(device),
    )

    command = " ".join(shlex.quote(arg) for arg in sys.argv)
    message_edges_per_epoch = graph.total_edges * config.model.num_layers

    print(f"Run directory: {run_dir}", flush=True)
    print(f"Backend: {backend_label(device)}", flush=True)
    print(f"Device: {hardware.get('device_name', 'unknown')}", flush=True)
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
        optimizer.zero_grad(set_to_none=True)
        reset_peak_memory_stats(device)
        synchronize_device(device)
        epoch_start = time.perf_counter()

        with autocast_context(device, config.run.precision):
            logits = model(graph.features, graph.relations)
            train_logits = logits.index_select(0, graph.train_index)
            loss = F.cross_entropy(train_logits, graph.train_labels)

        detached_logits = train_logits.detach()
        accuracy = (
            detached_logits.argmax(dim=1).eq(graph.train_labels).float().mean().item()
        )
        loss_value = float(loss.detach().item())

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        synchronize_device(device)
        epoch_seconds = time.perf_counter() - epoch_start
        elapsed_seconds = time.perf_counter() - training_start
        memory_stats = read_memory_stats(device)

        record = {
            "epoch": epoch,
            "loss": loss_value,
            "train_accuracy": accuracy,
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
            "framework": "pytorch",
            "backend": backend_label(device),
            "device": str(device),
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
            int(graph.train_index.numel()),
        ),
        "autosize": autosize_summary,
        "config": asdict(config),
    }

    benchmark_utils.write_history_csv(run_dir / "history.csv", history)
    benchmark_utils.write_summary_json(run_dir / "summary.json", summary)
    print(f"Summary written to {run_dir / 'summary.json'}", flush=True)
    return run_dir


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    config = apply_cli_overrides(load_benchmark_config(args.config), args)
    validate_config(config, device)
    run_benchmark(config, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
