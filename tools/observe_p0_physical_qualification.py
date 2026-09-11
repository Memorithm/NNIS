#!/usr/bin/env python3
"""Read-only observer for long-running NNIS P0 physical qualification campaigns.

This tool is deliberately outside the qualification evidence path. It never mutates
benchmark state, never signals a process, and never changes promotion decisions. Its
only optional write is an atomic observer snapshot chosen by the caller.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Iterable

KIND = "nnis-p0-observer-snapshot-v1"
PROCESS_MARKERS = (
    "run_p0_physical_qualification_bundle.py",
    "run_tinyllama_massive_campaign.py",
    "llama_f16_massive_abba",
)
ARTIFACT_NAMES = (
    "campaign_01.json",
    "campaign_02.json",
    "consensus.json",
    "parity_record.json",
    "nnml1-multi-model-parity-suite.json",
    "P0_PHYSICAL_QUALIFICATION.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Observe an NNIS P0 physical qualification without perturbing it."
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _proc_stat(pid: int) -> tuple[int, int, int] | None:
    raw = _read_text(Path("/proc") / str(pid) / "stat")
    if raw is None:
        return None
    close = raw.rfind(")")
    if close < 0:
        return None
    fields = raw[close + 2 :].split()
    try:
        ppid = int(fields[1])
        utime = int(fields[11])
        stime = int(fields[12])
        start_ticks = int(fields[19])
    except (IndexError, ValueError):
        return None
    return ppid, utime + stime, start_ticks


def _rss_bytes(pid: int) -> int | None:
    raw = _read_text(Path("/proc") / str(pid) / "status")
    if raw is None:
        return None
    for line in raw.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1]) * 1024
                except ValueError:
                    return None
    return None


def iter_relevant_processes() -> Iterable[dict[str, Any]]:
    try:
        clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        uptime = float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, ValueError, KeyError):
        return []

    rows: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
        except (OSError, UnicodeDecodeError):
            continue
        marker = next((value for value in PROCESS_MARKERS if value in command), None)
        if marker is None:
            continue
        stat = _proc_stat(pid)
        if stat is None:
            continue
        ppid, cpu_ticks, start_ticks = stat
        elapsed = max(0.0, uptime - (start_ticks / clock_ticks))
        rows.append(
            {
                "pid": pid,
                "ppid": ppid,
                "role": marker,
                "elapsed_seconds": round(elapsed, 3),
                "cpu_seconds": round(cpu_ticks / clock_ticks, 3),
                "rss_bytes": _rss_bytes(pid),
                "command": command,
            }
        )
    return sorted(rows, key=lambda row: row["pid"])


def gpu_process_memory() -> dict[int, int]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if completed.returncode != 0:
        return {}
    result: dict[int, int] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            result[int(fields[0])] = int(fields[1])
        except ValueError:
            continue
    return result


def artifact_inventory(work_dir: Path, expected_head: str | None) -> list[dict[str, Any]]:
    candidates: list[Path] = []
    if expected_head:
        candidates.extend(
            [
                work_dir / "tinyllama" / f"runs-{expected_head[:12]}" / "campaign_01.json",
                work_dir / "tinyllama" / f"runs-{expected_head[:12]}" / "campaign_02.json",
                work_dir / "tinyllama" / f"runs-{expected_head[:12]}" / "consensus.json",
                work_dir / "tinyllama" / f"runs-{expected_head[:12]}" / "parity_record.json",
            ]
        )
    else:
        for runs in sorted((work_dir / "tinyllama").glob("runs-*")):
            candidates.extend(runs / name for name in ARTIFACT_NAMES[:4])
    candidates.extend(
        [
            work_dir / "nnml1-multi-model-parity-suite.json",
            work_dir / "P0_PHYSICAL_QUALIFICATION.json",
        ]
    )

    result: list[dict[str, Any]] = []
    for path in candidates:
        if not path.is_file():
            continue
        stat = path.stat()
        result.append(
            {
                "name": path.name,
                "path": str(path),
                "bytes": stat.st_size,
                "mtime_unix_seconds": stat.st_mtime,
            }
        )
    return result


def classify_state(processes: list[dict[str, Any]], artifacts: list[dict[str, Any]]) -> str:
    names = {artifact["name"] for artifact in artifacts}
    roles = {process["role"] for process in processes}
    if "P0_PHYSICAL_QUALIFICATION.json" in names:
        return "bundle_complete"
    if "nnml1-multi-model-parity-suite.json" in names:
        return "parity_suite_complete"
    if "consensus.json" in names and "parity_record.json" in names:
        return "tinyllama_complete"
    if "llama_f16_massive_abba" in roles:
        return "tinyllama_benchmark_active"
    if "run_tinyllama_massive_campaign.py" in roles:
        return "tinyllama_launcher_active"
    if "run_p0_physical_qualification_bundle.py" in roles:
        return "p0_launcher_active"
    return "inactive_or_finished_without_final_bundle"


def build_snapshot(work_dir: Path, expected_head: str | None) -> dict[str, Any]:
    processes = list(iter_relevant_processes())
    gpu_memory = gpu_process_memory()
    for process in processes:
        process["gpu_memory_mib"] = gpu_memory.get(process["pid"])
    artifacts = artifact_inventory(work_dir, expected_head)
    return {
        "schema_version": 1,
        "kind": KIND,
        "observer_only": True,
        "qualification_authority": False,
        "timestamp_unix_seconds": time.time(),
        "work_dir": str(work_dir),
        "expected_head": expected_head,
        "state": classify_state(processes, artifacts),
        "processes": processes,
        "artifacts": artifacts,
        "claim_boundary": (
            "read-only operational observation only; this snapshot is not qualification "
            "evidence and cannot authorize promotion or replace benchmark artifacts"
        ),
    }


def atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def self_test() -> None:
    active = [{"role": "llama_f16_massive_abba"}]
    assert classify_state(active, []) == "tinyllama_benchmark_active"
    assert classify_state([], [{"name": "consensus.json"}, {"name": "parity_record.json"}]) == "tinyllama_complete"
    assert classify_state([], [{"name": "P0_PHYSICAL_QUALIFICATION.json"}]) == "bundle_complete"
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        head = "a" * 40
        run = work / "tinyllama" / f"runs-{head[:12]}"
        run.mkdir(parents=True)
        (run / "campaign_01.json").write_text("{}\n", encoding="utf-8")
        inventory = artifact_inventory(work, head)
        assert [item["name"] for item in inventory] == ["campaign_01.json"]
        output = work / "observer.json"
        atomic_write_json(output, {"ok": True})
        assert json.loads(output.read_text(encoding="utf-8")) == {"ok": True}
    print("observe_p0_physical_qualification self-test: PASS")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.work_dir is None:
        raise SystemExit("--work-dir is required unless --self-test is used")
    work_dir = args.work_dir.expanduser().resolve()
    snapshot = build_snapshot(work_dir, args.expected_head)
    if args.output is not None:
        atomic_write_json(args.output.expanduser().resolve(), snapshot)
    print(json.dumps(snapshot, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
