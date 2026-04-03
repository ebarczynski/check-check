from __future__ import annotations

import csv
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


@dataclass
class SizingConfig:
    enabled: bool = False
    target_memory_gb: float | None = None
    target_memory_fraction: float | None = 0.85
    tolerance: float = 0.08
    max_probes: int = 6


@dataclass
class AutosizeProbe:
    probe: int
    scale: float
    num_nodes: int
    edges_per_relation: int
    measured_memory_gb: float | None
    epoch_seconds: float | None
    oom: bool
    note: str | None = None


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_config_path(path: Path) -> Path:
    base_candidates = [path]
    if path.suffix != ".json":
        base_candidates.append(path.with_suffix(".json"))

    candidates: list[Path] = []
    root = project_root()
    for candidate in base_candidates:
        candidates.append(candidate)
        if not candidate.is_absolute():
            candidates.append(root / candidate)
            if len(candidate.parts) == 1:
                candidates.append(root / "configs" / candidate.name)

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.is_file():
            return candidate

    preset_dir = root / "configs"
    available_presets = []
    if preset_dir.is_dir():
        available_presets = sorted(entry.name for entry in preset_dir.glob("*.json"))

    attempted = ", ".join(str(candidate) for candidate in unique_candidates)
    available = ", ".join(available_presets) if available_presets else "none found"
    raise FileNotFoundError(
        "Could not find benchmark config. "
        f"Tried: {attempted}. "
        f"Available presets in configs/: {available}."
    )


def slugify(value: str) -> str:
    collapsed = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return collapsed.strip("-") or "run"


def create_run_dir(results_dir: str | Path, run_name: str, backend: str) -> Path:
    root = Path(results_dir)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"{timestamp}_{slugify(run_name)}_{backend}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def precision_bytes(precision: str) -> int:
    return 4 if precision == "fp32" else 2


def effective_memory_usage_gb(memory_stats: dict[str, float | None]) -> float | None:
    for key in (
        "reserved_gb",
        "driver_allocated_gb",
        "allocated_gb",
        "peak_reserved_gb",
        "peak_allocated_gb",
    ):
        value = memory_stats.get(key)
        if value is not None:
            return value
    return None


def scale_graph_config(graph_config: Any, scale: float) -> Any:
    bounded_scale = max(scale, 0.05)
    num_nodes = max(1_024, int(round(graph_config.num_nodes * bounded_scale)))
    edges_per_relation = max(
        1_024,
        int(round(graph_config.edges_per_relation * bounded_scale)),
    )
    return replace(
        graph_config,
        num_nodes=num_nodes,
        edges_per_relation=edges_per_relation,
    )


def graph_scale_factor(base_graph_config: Any, scaled_graph_config: Any) -> float:
    node_ratio = scaled_graph_config.num_nodes / max(base_graph_config.num_nodes, 1)
    edge_ratio = scaled_graph_config.edges_per_relation / max(
        base_graph_config.edges_per_relation,
        1,
    )
    return (node_ratio + edge_ratio) / 2.0


def resolve_target_memory_gb(
    sizing_config: SizingConfig,
    available_memory_gb: float | None,
) -> tuple[float, list[str]]:
    notes: list[str] = []
    if sizing_config.target_memory_gb is not None:
        target_memory_gb = sizing_config.target_memory_gb
        notes.append("target_memory_gb")
    elif (
        sizing_config.target_memory_fraction is not None
        and available_memory_gb is not None
    ):
        target_memory_gb = available_memory_gb * sizing_config.target_memory_fraction
        notes.append("target_memory_fraction")
    else:
        raise ValueError(
            "autosize requires either target_memory_gb or target_memory_fraction with a known available memory capacity"
        )

    if available_memory_gb is not None and target_memory_gb >= available_memory_gb:
        target_memory_gb = available_memory_gb * 0.98
        notes.append("clamped_to_available_memory")

    return target_memory_gb, notes


def estimate_epoch_telemetry(
    graph_config: Any,
    model_config: Any,
    precision: str,
    *,
    feature_storage_bytes: int,
    parameter_storage_bytes: int,
    index_dtype_bytes: int,
    label_dtype_bytes: int,
) -> dict[str, float]:
    activation_bytes = precision_bytes(precision)
    num_nodes = float(graph_config.num_nodes)
    num_relations = float(graph_config.num_relations)
    total_edges = float(graph_config.num_relations * graph_config.edges_per_relation)
    train_nodes = float(max(1, int(graph_config.num_nodes * graph_config.train_fraction)))
    hidden_dim = float(model_config.hidden_dim)
    inner_dim = float(model_config.hidden_dim * model_config.ffn_multiplier)
    num_layers = float(model_config.num_layers)
    num_classes = float(graph_config.num_classes)
    input_dim = float(graph_config.input_dim)

    parameter_elements = (
        input_dim * hidden_dim
        + num_layers
        * (
            num_relations * hidden_dim * hidden_dim
            + hidden_dim * hidden_dim
            + hidden_dim * inner_dim
            + inner_dim * hidden_dim
            + 4.0 * hidden_dim
        )
        + 2.0 * hidden_dim
        + hidden_dim * num_classes
        + num_classes
    )
    graph_static_bytes = (
        num_nodes * input_dim * feature_storage_bytes
        + num_nodes * label_dtype_bytes
        + train_nodes * (index_dtype_bytes + label_dtype_bytes)
        + total_edges * (2.0 * index_dtype_bytes + feature_storage_bytes)
    )
    parameter_static_bytes = parameter_elements * parameter_storage_bytes

    forward_ops = (
        num_nodes * 2.0 * input_dim * hidden_dim
        + num_layers
        * (
            total_edges * (2.0 * hidden_dim * hidden_dim + 2.0 * hidden_dim)
            + num_nodes
            * (
                2.0 * hidden_dim * hidden_dim
                + 4.0 * hidden_dim * inner_dim
                + 16.0 * hidden_dim
            )
        )
        + num_nodes * 2.0 * hidden_dim * num_classes
    )
    training_ops = forward_ops * 3.0

    forward_bytes = (
        num_nodes * (input_dim * feature_storage_bytes + hidden_dim * activation_bytes)
        + num_layers
        * (
            total_edges
            * (
                4.0 * hidden_dim * activation_bytes
                + feature_storage_bytes
                + 2.0 * index_dtype_bytes
            )
            + num_relations * hidden_dim * hidden_dim * parameter_storage_bytes
            + num_nodes * ((10.0 * hidden_dim + 3.0 * inner_dim) * activation_bytes)
            + (hidden_dim * hidden_dim + 2.0 * hidden_dim * inner_dim + 4.0 * hidden_dim)
            * parameter_storage_bytes
        )
        + num_nodes * ((2.0 * hidden_dim + num_classes) * activation_bytes)
        + train_nodes * (index_dtype_bytes + label_dtype_bytes)
        + (hidden_dim * num_classes + num_classes) * parameter_storage_bytes
    )
    training_bytes = forward_bytes * 3.0

    return {
        "estimated_graph_static_bytes": graph_static_bytes,
        "estimated_parameter_static_bytes": parameter_static_bytes,
        "estimated_forward_bytes_per_epoch": forward_bytes,
        "estimated_training_bytes_per_epoch": training_bytes,
        "estimated_forward_ops_per_epoch": forward_ops,
        "estimated_training_ops_per_epoch": training_ops,
    }


def attach_epoch_telemetry(
    record: dict[str, Any],
    epoch_estimates: dict[str, float],
) -> dict[str, Any]:
    epoch_seconds = max(float(record["epoch_seconds"]), 1e-9)
    record.update(epoch_estimates)
    record["estimated_bandwidth_gbps"] = (
        epoch_estimates["estimated_training_bytes_per_epoch"] / epoch_seconds / 1e9
    )
    record["estimated_tops"] = (
        epoch_estimates["estimated_training_ops_per_epoch"] / epoch_seconds / 1e12
    )
    return record


def steady_state_records(
    history: list[dict[str, Any]],
    warmup_epochs: int,
) -> list[dict[str, Any]]:
    steady = [record for record in history if record["epoch"] > warmup_epochs]
    return steady if steady else history


def summarize_history(
    history: list[dict[str, Any]],
    warmup_epochs: int,
    train_nodes: int,
) -> dict[str, Any]:
    steady = steady_state_records(history, warmup_epochs)
    epoch_seconds = [record["epoch_seconds"] for record in steady]
    message_rates = [record["message_edge_updates_per_sec"] for record in steady]
    accuracy_values = [record["train_accuracy"] for record in steady]
    peak_values = [
        record["peak_memory_gb"]
        for record in history
        if record.get("peak_memory_gb") is not None
    ]

    summary = {
        "epochs_completed": history[-1]["epoch"],
        "warmup_epochs_excluded": warmup_epochs,
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
        "train_nodes": train_nodes,
    }

    metric_prefixes = {
        "estimated_bandwidth_gbps": "steady_state_estimated_bandwidth_gbps",
        "estimated_tops": "steady_state_estimated_tops",
        "effective_memory_gb": "steady_state_effective_memory_gb",
    }
    for source_key, prefix in metric_prefixes.items():
        values = [record[source_key] for record in steady if record.get(source_key) is not None]
        if values:
            summary[f"{prefix}_median"] = statistics.median(values)
            summary[f"{prefix}_mean"] = statistics.fmean(values)

    constant_metrics = (
        "estimated_graph_static_bytes",
        "estimated_parameter_static_bytes",
        "estimated_forward_bytes_per_epoch",
        "estimated_training_bytes_per_epoch",
        "estimated_forward_ops_per_epoch",
        "estimated_training_ops_per_epoch",
    )
    for key in constant_metrics:
        value = history[-1].get(key)
        if value is not None:
            summary[key] = value

    return summary


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
    effective_memory = record.get("effective_memory_gb")
    effective_memory_fragment = (
        f"mem={effective_memory:.2f}GB" if effective_memory is not None else "mem=n/a"
    )
    peak_memory = record.get("peak_memory_gb")
    peak_fragment = (
        f"peak_mem={peak_memory:.2f}GB" if peak_memory is not None else "peak_mem=n/a"
    )
    bandwidth = record.get("estimated_bandwidth_gbps")
    bandwidth_fragment = (
        f"bw={bandwidth:.2f}GB/s" if bandwidth is not None else "bw=n/a"
    )
    tops = record.get("estimated_tops")
    tops_fragment = f"tops={tops:.4f}" if tops is not None else "tops=n/a"
    return (
        f"epoch={record['epoch']:04d} "
        f"loss={record['loss']:.4f} "
        f"acc={record['train_accuracy']:.4f} "
        f"epoch_s={record['epoch_seconds']:.2f} "
        f"edge_updates/s={record['message_edge_updates_per_sec'] / 1e6:.2f}M "
        f"{bandwidth_fragment} "
        f"{tops_fragment} "
        f"elapsed_s={record['elapsed_seconds']:.2f} "
        f"{effective_memory_fragment} "
        f"{peak_fragment}"
    )


def auto_size_graph(
    base_graph_config: Any,
    sizing_config: SizingConfig,
    available_memory_gb: float | None,
    probe_fn: Callable[[Any, float, int], AutosizeProbe],
) -> tuple[Any, dict[str, Any]]:
    if not sizing_config.enabled:
        return base_graph_config, {
            "enabled": False,
            "target_memory_gb": sizing_config.target_memory_gb,
            "available_memory_gb": available_memory_gb,
            "probe_history": [],
        }

    target_memory_gb, notes = resolve_target_memory_gb(
        sizing_config,
        available_memory_gb,
    )

    probe_history: list[AutosizeProbe] = []
    best_probe: AutosizeProbe | None = None
    best_error: float | None = None
    lower_bound: tuple[float, float] | None = None
    upper_bound: tuple[float, float | None] | None = None
    attempted_shapes: set[tuple[int, int]] = set()
    candidate_scale = 1.0

    for probe_index in range(1, sizing_config.max_probes + 1):
        candidate_graph = scale_graph_config(base_graph_config, candidate_scale)
        shape_key = (
            candidate_graph.num_nodes,
            candidate_graph.edges_per_relation,
        )
        if shape_key in attempted_shapes:
            break
        attempted_shapes.add(shape_key)

        actual_scale = graph_scale_factor(base_graph_config, candidate_graph)
        probe = probe_fn(candidate_graph, actual_scale, probe_index)
        probe_history.append(probe)

        if not probe.oom and probe.measured_memory_gb is not None:
            error = abs(probe.measured_memory_gb - target_memory_gb)
            if best_probe is None or best_error is None or error < best_error:
                best_probe = probe
                best_error = error

            if error <= target_memory_gb * sizing_config.tolerance:
                break

            if probe.measured_memory_gb < target_memory_gb:
                lower_bound = (actual_scale, probe.measured_memory_gb)
            else:
                upper_bound = (actual_scale, probe.measured_memory_gb)
        else:
            upper_bound = (actual_scale, probe.measured_memory_gb)

        next_scale = choose_next_scale(
            actual_scale=actual_scale,
            measured_memory_gb=probe.measured_memory_gb,
            target_memory_gb=target_memory_gb,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        if next_scale is None:
            break
        candidate_scale = next_scale

    if best_probe is None:
        raise RuntimeError(
            "autosize could not find a non-OOM probe for the requested memory budget"
        )

    final_graph_config = scale_graph_config(base_graph_config, best_probe.scale)
    autosize_summary = {
        "enabled": True,
        "available_memory_gb": available_memory_gb,
        "target_memory_gb": target_memory_gb,
        "target_notes": notes,
        "tolerance": sizing_config.tolerance,
        "max_probes": sizing_config.max_probes,
        "base_graph": asdict(base_graph_config),
        "final_graph": asdict(final_graph_config),
        "final_scale": graph_scale_factor(base_graph_config, final_graph_config),
        "final_measured_memory_gb": best_probe.measured_memory_gb,
        "probe_history": [asdict(probe) for probe in probe_history],
    }
    return final_graph_config, autosize_summary


def choose_next_scale(
    *,
    actual_scale: float,
    measured_memory_gb: float | None,
    target_memory_gb: float,
    lower_bound: tuple[float, float] | None,
    upper_bound: tuple[float, float | None] | None,
) -> float | None:
    if lower_bound is not None and upper_bound is not None:
        lower_scale = lower_bound[0]
        upper_scale = upper_bound[0]
        next_scale = math.sqrt(lower_scale * upper_scale)
        if abs(next_scale - actual_scale) / max(actual_scale, 1e-6) < 0.03:
            return None
        return next_scale

    if measured_memory_gb is None or measured_memory_gb <= 0.0:
        return actual_scale * 0.5 if upper_bound is not None else actual_scale * 2.0

    ratio = target_memory_gb / measured_memory_gb
    if measured_memory_gb < target_memory_gb:
        next_scale = actual_scale * min(2.5, max(1.1, ratio))
    else:
        next_scale = actual_scale * max(0.35, min(0.9, ratio))

    if abs(next_scale - actual_scale) / max(actual_scale, 1e-6) < 0.03:
        return None
    return next_scale