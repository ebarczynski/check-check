from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SORT_KEY_MAP = {
    "steady_state_message_edge_updates_per_sec_median": "edge_updates_per_sec",
    "steady_state_estimated_bandwidth_gbps_median": "estimated_bandwidth_gbps",
    "steady_state_estimated_tops_median": "estimated_tops",
    "steady_state_epoch_sec_median": "epoch_sec",
    "peak_memory_gb": "peak_memory_gb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare summary.json outputs from multiple benchmark runs."
    )
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument(
        "--sort-by",
        choices=(
            "steady_state_message_edge_updates_per_sec_median",
            "steady_state_estimated_bandwidth_gbps_median",
            "steady_state_estimated_tops_median",
            "steady_state_epoch_sec_median",
            "peak_memory_gb",
        ),
        default="steady_state_message_edge_updates_per_sec_median",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        help="Optional path to also write a CSV table.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    workload = payload["workload"]
    hardware = payload["hardware"]
    run = payload["run"]

    return {
        "summary_path": str(path),
        "run": run["name"],
        "framework": run.get("framework", "pytorch"),
        "backend": run["backend"],
        "device_name": hardware.get("device_name", "unknown"),
        "precision": run["precision"],
        "num_nodes": workload["num_nodes"],
        "total_edges": workload["total_edges"],
        "hidden_dim": workload["hidden_dim"],
        "num_layers": workload["num_layers"],
        "epoch_sec": metrics["steady_state_epoch_sec_median"],
        "edge_updates_per_sec": metrics[
            "steady_state_message_edge_updates_per_sec_median"
        ],
        "estimated_bandwidth_gbps": metrics.get(
            "steady_state_estimated_bandwidth_gbps_median"
        ),
        "estimated_tops": metrics.get("steady_state_estimated_tops_median"),
        "peak_memory_gb": metrics["peak_memory_gb"],
        "training_duration_sec": metrics["training_duration_sec"],
    }


def format_number(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def rows_to_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Run",
        "Framework",
        "Backend",
        "Device",
        "Precision",
        "Nodes",
        "Edges",
        "Hidden",
        "Layers",
        "Median epoch s",
        "Median M edge updates/s",
        "Median GB/s",
        "Median TOPS",
        "Peak GB",
        "Duration s",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["run"],
                    row["framework"],
                    row["backend"],
                    row["device_name"],
                    row["precision"],
                    format_number(row["num_nodes"], 0),
                    format_number(row["total_edges"], 0),
                    format_number(row["hidden_dim"], 0),
                    format_number(row["num_layers"], 0),
                    format_number(row["epoch_sec"]),
                    format_number(row["edge_updates_per_sec"] / 1e6),
                    format_number(row["estimated_bandwidth_gbps"]),
                    format_number(row["estimated_tops"], 4),
                    format_number(row["peak_memory_gb"]),
                    format_number(row["training_duration_sec"]),
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sort_value(row: dict[str, Any], sort_key: str, reverse: bool) -> float:
    value = row.get(sort_key)
    if value is None:
        return float("-inf") if reverse else float("inf")
    return float(value)


def main() -> int:
    args = parse_args()
    rows = [load_summary(path) for path in args.summaries]

    sort_key = SORT_KEY_MAP[args.sort_by]
    reverse = args.sort_by != "steady_state_epoch_sec_median"
    rows.sort(key=lambda row: sort_value(row, sort_key, reverse), reverse=reverse)

    print(rows_to_markdown(rows))
    if args.csv_out:
        write_csv(args.csv_out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
