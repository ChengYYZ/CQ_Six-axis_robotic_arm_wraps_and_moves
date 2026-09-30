from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TextIO


SUMMARY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("completed_packages", "Completed batch candidate #"),
    ("motion_stops", "Batch motion stopped at candidate #"),
    ("controller_conf_rejections", "CONTROLLER_ERROR_EVENT"),
    ("controller_singularity_rejections", "CONTROLLER_ERROR_EVENT"),
    ("large_pickup_rotations_rejected", "PLAN_REJECTION_SUMMARY"),
    ("unused_cup_collision_rejections", "PLAN_REJECTION_SUMMARY"),
    ("workspace_rejections", "PLAN_REJECTION_SUMMARY"),
    ("controller_trial_limits", "controller reachability trials"),
    ("verified_d_fallback_trials", "Trying verified placement rotation"),
    ("degraded_placements", "placement_quality=degraded"),
    ("barcode_timeouts", "No new top/front barcode matched"),
    ("empty_scene_returns", "No package detected in the fresh camera analysis"),
    ("prefetched_scenes", "Next-scene analysis #"),
)


class _TeeStream:
    """Mirror console text while writing timestamped, thread-labelled complete lines."""

    def __init__(
        self,
        console: TextIO,
        log_file: TextIO,
        lock: threading.RLock,
        stream_name: str,
    ) -> None:
        self._console = console
        self._log_file = log_file
        self._lock = lock
        self._stream_name = stream_name
        self._pending: dict[int, str] = {}

    @property
    def encoding(self):
        return getattr(self._console, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._console, "errors", "replace")

    def isatty(self) -> bool:
        return bool(getattr(self._console, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._console.fileno()

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            text = str(text)
        thread_id = threading.get_ident()
        with self._lock:
            self._console.write(text)
            buffered = self._pending.get(thread_id, "") + text
            while "\n" in buffered:
                line, buffered = buffered.split("\n", 1)
                self._write_log_line(line)
            self._pending[thread_id] = buffered
            self._log_file.flush()
        return len(text)

    def flush(self) -> None:
        with self._lock:
            self._console.flush()
            self._log_file.flush()

    def flush_pending(self) -> None:
        with self._lock:
            for thread_id, text in list(self._pending.items()):
                if text:
                    self._write_log_line(text, thread_id=thread_id)
            self._pending.clear()
            self._log_file.flush()

    def _write_log_line(self, line: str, *, thread_id: int | None = None) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        if thread_id is None:
            thread_name = threading.current_thread().name
        else:
            thread_name = next(
                (thread.name for thread in threading.enumerate() if thread.ident == thread_id),
                f"thread-{thread_id}",
            )
        self._log_file.write(
            f"[{timestamp}] [{thread_name}] [{self._stream_name}] {line}\n"
        )

    def __getattr__(self, name: str):
        return getattr(self._console, name)


@dataclass
class RunLogSession:
    log_path: Path
    summary_path: Path
    log_file: TextIO
    original_stdout: TextIO
    original_stderr: TextIO
    stdout_tee: _TeeStream
    stderr_tee: _TeeStream
    create_summary: bool = True
    started_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    closed: bool = False

    def close(self, exit_code: int | None = None) -> None:
        if self.closed:
            return
        self.closed = True
        duration_s = (datetime.now().astimezone() - self.started_at).total_seconds()
        print(
            f"Run log closing: exit_code={exit_code if exit_code is not None else 'unknown'} "
            f"duration={duration_s:.1f}s."
        )
        self.stdout_tee.flush_pending()
        self.stderr_tee.flush_pending()
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.log_file.flush()
        self.log_file.close()
        summary_error: Exception | None = None
        trend_path = self.log_path.parent / "trend_summary.md"
        if self.create_summary:
            try:
                analyze_runtime_log(self.log_path, self.summary_path)
                analyze_runtime_log_directory(self.log_path.parent, trend_path)
            except Exception as exc:  # Logging must never mask the robot process result.
                summary_error = exc
        self.original_stdout.write(f"Complete run log: {self.log_path}\n")
        if self.create_summary and summary_error is None:
            self.original_stdout.write(f"Run diagnostic summary: {self.summary_path}\n")
            self.original_stdout.write(f"Cross-run trend summary: {trend_path}\n")
        elif summary_error is not None:
            self.original_stderr.write(
                f"Warning: failed to create run diagnostic summary: {summary_error}\n"
            )
        self.original_stdout.flush()
        self.original_stderr.flush()


_ACTIVE_SESSION: RunLogSession | None = None


def _redacted_argv(argv: list[str]) -> list[str]:
    redacted = list(argv)
    sensitive_markers = ("password", "token", "secret", "api-key", "api_key")
    index = 0
    while index < len(redacted):
        value = redacted[index]
        lowered = value.lower()
        if any(marker in lowered for marker in sensitive_markers):
            if "=" in value:
                redacted[index] = value.split("=", 1)[0] + "=<redacted>"
            elif index + 1 < len(redacted):
                redacted[index + 1] = "<redacted>"
                index += 1
        index += 1
    return redacted


def _source_fingerprints() -> list[str]:
    root = Path(__file__).resolve().parent
    sources = (
        root / "surface_cluster_grasp.py",
        root / "project0714_calib" / "xcore_robot.py",
        root / "project0714_grasp" / "waybill_inspection.py",
    )
    fingerprints: list[str] = []
    for source in sources:
        try:
            digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
            fingerprints.append(f"{source.name}: sha256={digest}")
        except OSError as exc:
            fingerprints.append(f"{source.name}: unavailable ({exc})")
    return fingerprints


def start_run_log(
    log_dir: str | Path,
    *,
    argv: list[str] | None = None,
    create_summary: bool = True,
) -> RunLogSession:
    global _ACTIVE_SESSION
    if _ACTIVE_SESSION is not None and not _ACTIVE_SESSION.closed:
        return _ACTIVE_SESSION
    directory = Path(log_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone()
    stem = f"surface_grasp_{now.strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}"
    log_path = directory / f"{stem}.log"
    summary_path = directory / f"{stem}.summary.md"
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    lock = threading.RLock()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout_tee = _TeeStream(original_stdout, log_file, lock, "STDOUT")
    stderr_tee = _TeeStream(original_stderr, log_file, lock, "STDERR")
    session = RunLogSession(
        log_path=log_path,
        summary_path=summary_path,
        log_file=log_file,
        original_stdout=original_stdout,
        original_stderr=original_stderr,
        stdout_tee=stdout_tee,
        stderr_tee=stderr_tee,
        create_summary=create_summary,
        started_at=now,
    )
    _ACTIVE_SESSION = session
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee
    print("=" * 88)
    print(f"Surface grasp run started: {now.isoformat(timespec='seconds')}")
    print(f"PID: {os.getpid()}")
    print(f"Working directory: {Path.cwd()}")
    print(f"Python: {sys.version.replace(os.linesep, ' ')}")
    print(f"Command arguments: {_redacted_argv(argv if argv is not None else sys.argv)!r}")
    print("Source fingerprints:")
    for fingerprint in _source_fingerprints():
        print(f"  {fingerprint}")
    print(f"Complete run log: {log_path}")
    print("=" * 88)
    return session


def close_active_run_log(exit_code: int | None = None) -> None:
    global _ACTIVE_SESSION
    if _ACTIVE_SESSION is None:
        return
    _ACTIVE_SESSION.close(exit_code)
    _ACTIVE_SESSION = None


def analyze_runtime_log(log_path: str | Path, output_path: str | Path | None = None) -> Path:
    source = Path(log_path).resolve()
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else source.with_suffix(".summary.md")
    )
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    counts = _event_counts(lines)
    noteworthy = [
        line
        for line in lines
        if any(
            marker in line.lower()
            for marker in (
                "warning:",
                "failed",
                "error:",
                "rejected",
                "motion stopped",
                "remains attached",
            )
        )
    ][-30:]
    recommendations: list[str] = []
    if counts["motion_stops"]:
        recommendations.append("优先检查带载动作停止点及其前一条控制器命令。")
    if counts["controller_conf_rejections"]:
        recommendations.append("统计 -50021 集中的候选区域、吸盘和姿态，继续前移几何过滤。")
    if counts["controller_singularity_rejections"]:
        recommendations.append("统计 -50102 前后的路径段，确认是否可由稳定关节分支替代。")
    if counts["barcode_timeouts"]:
        recommendations.append("核对条码消息序号、网络到达时间和包裹释放时间窗口。")
    if not recommendations:
        recommendations.append("本次未命中已知高风险模式，仍需结合完整时序检查节拍和漏检。")

    rows = "\n".join(
        f"| `{key}` | {counts[key]} |" for key, _pattern in SUMMARY_PATTERNS
    )
    noteworthy_block = (
        "\n".join(f"- `{line}`" for line in noteworthy)
        if noteworthy
        else "- 未提取到明显异常行。"
    )
    recommendation_block = "\n".join(f"- {item}" for item in recommendations)
    report = f"""# Surface grasp run diagnostic summary

- Source log: `{source}`
- Total lines: {len(lines)}

## Event counts

| Event | Count |
|---|---:|
{rows}

## Suggested review focus

{recommendation_block}

## Last noteworthy lines

{noteworthy_block}
"""
    destination.write_text(report, encoding="utf-8")
    return destination


def _event_counts(lines: list[str]) -> Counter:
    counts = Counter(
        {
            key: sum(pattern in line for line in lines)
            for key, pattern in SUMMARY_PATTERNS
            if key
            not in {
                "controller_conf_rejections",
                "controller_singularity_rejections",
                "large_pickup_rotations_rejected",
                "unused_cup_collision_rejections",
                "workspace_rejections",
            }
        }
    )
    # Controller codes are counted only from one structured event emitted for
    # one physical controller attempt. Raw exception text can repeat the same
    # code in nested error messages and must not inflate the trend.
    structured_conf = sum(
        "CONTROLLER_ERROR_EVENT" in line and "code=-50021" in line
        for line in lines
    )
    structured_singularity = sum(
        "CONTROLLER_ERROR_EVENT" in line and "code=-50102" in line
        for line in lines
    )
    has_structured_controller_events = any(
        "CONTROLLER_ERROR_EVENT" in line for line in lines
    )
    if has_structured_controller_events:
        counts["controller_conf_rejections"] = structured_conf
        counts["controller_singularity_rejections"] = structured_singularity
    else:
        # Legacy logs predate structured events. Each operational failure line
        # represents one caught command; exclude the later "Last error" echo.
        counts["controller_conf_rejections"] = sum(
            "-50021" in line and "Last controller error" not in line
            for line in lines
        )
        counts["controller_singularity_rejections"] = sum(
            "-50102" in line and "Last controller error" not in line
            for line in lines
        )

    branch_rejections: Counter = Counter()
    for line in lines:
        if "PLAN_REJECTION_SUMMARY" not in line:
            continue
        match = re.search(r"\breasons=([^\s]+)", line)
        if match is None or match.group(1) == "none":
            continue
        for item in match.group(1).split(","):
            reason, separator, value = item.partition("=")
            if not separator:
                continue
            try:
                branch_rejections[reason] += int(value)
            except ValueError:
                continue
    if any("PLAN_REJECTION_SUMMARY" in line for line in lines):
        counts["large_pickup_rotations_rejected"] = branch_rejections[
            "pickup_rotation"
        ]
        counts["unused_cup_collision_rejections"] = branch_rejections[
            "unused_cup_collision"
        ]
        counts["workspace_rejections"] = (
            branch_rejections["pickup_workspace"]
            + branch_rejections["placement_envelope"]
        )
    else:
        counts["large_pickup_rotations_rejected"] = sum(
            "excessive A*-to-pickup rotation" in line for line in lines
        )
        counts["unused_cup_collision_rejections"] = sum(
            "unused-cup collision risk" in line for line in lines
        )
        counts["workspace_rejections"] = sum(
            "outside configured TCP workspace" in line for line in lines
        )
    return counts


def analyze_runtime_log_directory(
    log_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    max_runs: int = 100,
) -> Path:
    directory = Path(log_dir).resolve()
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else directory / "trend_summary.md"
    )
    log_paths = sorted(directory.glob("surface_grasp_*.log"))[-max(1, int(max_runs)):]
    rows: list[str] = []
    totals: Counter = Counter()
    for log_path in log_paths:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        counts = _event_counts(lines)
        totals.update(counts)
        rows.append(
            "| {run} | {completed} | {stops} | {conf} | {singularity} | "
            "{rotation} | {collision} | {barcode} |".format(
                run=log_path.stem,
                completed=counts["completed_packages"],
                stops=counts["motion_stops"],
                conf=counts["controller_conf_rejections"],
                singularity=counts["controller_singularity_rejections"],
                rotation=counts["large_pickup_rotations_rejected"],
                collision=counts["unused_cup_collision_rejections"],
                barcode=counts["barcode_timeouts"],
            )
        )
    completed = totals["completed_packages"]
    motion_stops = totals["motion_stops"]
    stop_rate = 100.0 * motion_stops / max(1, completed + motion_stops)
    table_rows = "\n".join(rows) if rows else "| _no runs_ | 0 | 0 | 0 | 0 | 0 | 0 | 0 |"
    report = f"""# Surface grasp cross-run trend

- Runs included: {len(log_paths)}
- Completed packages: {completed}
- Motion stops: {motion_stops}
- Motion-stop share of completed/stopped attempts: {stop_rate:.1f}%

| Run | Completed | Motion stops | -50021 | -50102 | Large rotation rejected | Cup collision rejected | Barcode timeout |
|---|---:|---:|---:|---:|---:|---:|---:|
{table_rows}

This report is evidence for comparison, not an instruction to change robot motion automatically.
Review the corresponding complete log before modifying safety or reachability logic.
"""
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze one run log or a directory of run logs")
    parser.add_argument("log", help="Path to a complete .log file or runtime_logs directory")
    parser.add_argument("--output", help="Optional summary Markdown path")
    args = parser.parse_args()
    source = Path(args.log)
    result = (
        analyze_runtime_log_directory(source, args.output)
        if source.is_dir()
        else analyze_runtime_log(source, args.output)
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
