#!/usr/bin/env python3

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
import re
import shlex
import subprocess
import time
from pathlib import Path


TIMER_METADATA_COLUMNS = (
    "timer_runtime",
    "timer_runtime_version",
    "timer_engine",
    "timer_engine_version",
    "timer_host_system",
    "timer_host_machine",
)
TIMER_IDENTITY_COLUMNS = (
    "timer_dataset",
    "timer_genes",
    "timer_samples",
    "timer_design",
    "timer_contrast",
)
TIMER_PROVENANCE_COLUMNS = (*TIMER_METADATA_COLUMNS, *TIMER_IDENTITY_COLUMNS)


def parse_args():
    parser = argparse.ArgumentParser(description="Run reproducible DESeq2 module timing repetitions.")
    parser.add_argument("--dataset", required=True, help="Dataset label written to the timing table.")
    parser.add_argument("--workflow", required=True, type=Path, help="Dataset-specific Nextflow workflow.")
    parser.add_argument("--config", required=True, type=Path, help="Dataset-specific Nextflow config.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Prepared dataset directory.")
    parser.add_argument("--outdir", required=True, type=Path, help="Benchmark output directory.")
    parser.add_argument("--cpus", default="1,4", help="Comma-separated CPU settings.")
    parser.add_argument("--implementations", default="r,pydeseq2", help="Comma-separated implementations.")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=7,
        help="Measured repetitions after one warm-up (default: 7).",
    )
    parser.add_argument("--profile", default="docker", help="Nextflow profile.")
    parser.add_argument("--nextflow", default="nextflow", help="Nextflow executable.")
    parser.add_argument("--nxf-home", type=Path, help="Writable NXF_HOME directory.")
    parser.add_argument("--r-core-runtime", type=Path, help="Repeated R reference runtime TSV.")
    parser.add_argument("--r-env-bin", type=Path, help="R environment bin directory for the local profile.")
    parser.add_argument(
        "--pydeseq2-env-bin",
        type=Path,
        help="PyDESeq2 environment bin directory for the local profile.",
    )
    parser.add_argument(
        "--environment-yaml",
        type=Path,
        help="Execution environment YAML; defaults to environment.yml beside the example directories.",
    )
    return parser.parse_args()


def parse_duration_seconds(value):
    if value is None:
        return None
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    total = 0.0
    matched = False
    for amount, unit in re.findall(r"([0-9]*\.?[0-9]+)\s*(ms|[dhms])", text):
        matched = True
        total += float(amount) * {"d": 86400, "h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total if matched else None


def parse_size_bytes(value):
    if value is None:
        return None
    match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([KMGTPE]?B)?\s*", str(value), re.IGNORECASE)
    if not match:
        return None
    amount = float(match.group(1))
    unit = (match.group(2) or "B").upper()
    scale = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}.get(unit)
    return int(amount * scale) if scale is not None else None


def read_nextflow_metrics(trace_path, implementation):
    process_name = "DESEQ2_DIFFERENTIAL" if implementation == "r" else "PYDESEQ2_DIFFERENTIAL"
    if not trace_path.exists() or trace_path.stat().st_size == 0:
        return None, None
    with trace_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    selected = [row for row in rows if process_name in row.get("name", row.get("process", ""))]
    if not selected:
        return None, None
    elapsed = [parse_duration_seconds(row.get("realtime") or row.get("duration")) for row in selected]
    peak_rss = [parse_size_bytes(row.get("peak_rss")) for row in selected]
    elapsed = [value for value in elapsed if value is not None]
    peak_rss = [value for value in peak_rss if value is not None]
    return (sum(elapsed) if elapsed else None, max(peak_rss) if peak_rss else None)


def read_command_trace(work_dir):
    traces = list(work_dir.glob("**/.command.trace"))
    values = []
    for path in traces:
        fields = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                fields[key] = value
        if fields.get("realtime"):
            values.append(float(fields["realtime"]) / 1000)
    return max(values) if values else None


def read_pydeseq2_phases(output_dir):
    runtime_files = list(output_dir.glob("**/*.pydeseq2.runtime.tsv"))
    if len(runtime_files) != 1:
        raise ValueError(f"Expected one PyDESeq2 runtime TSV beneath {output_dir}")
    with runtime_files[0].open(newline="", encoding="utf-8") as handle:
        phases = {row["step"]: float(row["elapsed_seconds"]) for row in csv.DictReader(handle, delimiter="\t")}
    if "pydeseq2_core_deseq2_and_stats" in phases:
        core_seconds = phases["pydeseq2_core_deseq2_and_stats"]
        core_source = "module_runtime_tsv:pydeseq2_core_deseq2_and_stats"
    elif {"core_dds_deseq2", "stats_summary"}.issubset(phases):
        core_seconds = phases["core_dds_deseq2"] + phases["stats_summary"]
        core_source = "module_runtime_tsv:core_dds_deseq2+stats_summary"
    else:
        core_seconds = None
        core_source = None
    if core_seconds is None:
        raise ValueError(f"PyDESeq2 runtime TSV does not contain a supported core timing in {runtime_files[0]}")
    if "vst_dds_vst" in phases:
        vst_seconds = phases["vst_dds_vst"]
        vst_source = "module_runtime_tsv:vst_dds_vst"
    else:
        vst_seconds = phases.get("pydeseq2_vst")
        vst_source = "module_runtime_tsv:pydeseq2_vst" if vst_seconds is not None else None
    return core_seconds, vst_seconds, core_source, vst_source, runtime_files[0]


def validate_external_r_runtime_setup(profile, runtime_path, r_env_bin, pydeseq2_env_bin):
    if runtime_path is None:
        return
    if profile != "local":
        raise ValueError("--r-core-runtime requires --profile local")
    if r_env_bin is None or pydeseq2_env_bin is None:
        raise ValueError("--r-core-runtime requires both --r-env-bin and --pydeseq2-env-bin")
    if not r_env_bin.is_dir() or not pydeseq2_env_bin.is_dir():
        raise ValueError("--r-core-runtime environment bin paths must be existing directories")
    if r_env_bin != pydeseq2_env_bin:
        raise ValueError("--r-core-runtime requires both environment bin paths to resolve to the same directory")
    if not runtime_path.is_file():
        raise ValueError(f"R core-runtime TSV does not exist: {runtime_path}")


def load_r_runtime(runtime_path):
    with runtime_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"cpus", "repetition", "step", "elapsed_seconds", *TIMER_PROVENANCE_COLUMNS}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"R core-runtime TSV is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"R core-runtime TSV is empty: {runtime_path}")

    metadata = {}
    for column in TIMER_PROVENANCE_COLUMNS:
        values = {row[column].strip() for row in rows}
        if "" in values or len(values) != 1:
            raise ValueError(f"R core-runtime {column} must be non-empty and constant")
        metadata[column] = values.pop()
    if metadata["timer_runtime"] != "R" or metadata["timer_engine"] != "DESeq2":
        raise ValueError("R core-runtime timer runtime/engine must be R and DESeq2")

    seen = set()
    for row in rows:
        try:
            cpus = int(row["cpus"])
            repetition = int(row["repetition"])
            elapsed_seconds = float(row["elapsed_seconds"])
        except ValueError as exc:
            raise ValueError("R core-runtime CPU, repetition, and elapsed values must be numeric") from exc
        if cpus < 1 or repetition < 1 or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("R core-runtime CPU/repetition values must be positive and elapsed time non-negative")
        step = row["step"].strip()
        if not step:
            raise ValueError("R core-runtime step values must not be empty")
        row["step"] = step
        key = (cpus, repetition, step)
        if key in seen:
            raise ValueError("R core-runtime TSV contains duplicate (cpus, repetition, step) rows")
        seen.add(key)
    return rows, metadata


def validate_r_timer_metadata(metadata, versions, host_system, host_machine):
    expected = {
        "timer_runtime": "R",
        "timer_runtime_version": versions["r_version"],
        "timer_engine": "DESeq2",
        "timer_engine_version": versions["deseq2_version"],
        "timer_host_system": host_system,
        "timer_host_machine": host_machine,
    }
    mismatches = [column for column, value in expected.items() if metadata.get(column) != value]
    if mismatches:
        raise ValueError(
            "R core-runtime metadata does not match the R module session/current host: "
            f"{', '.join(mismatches)}"
        )


def validate_r_timer_identity(metadata, dataset, genes, samples, design, contrast):
    mismatches = []
    if metadata.get("timer_dataset", "").casefold() != dataset.casefold():
        mismatches.append("timer_dataset")
    for column, expected in (("timer_genes", genes), ("timer_samples", samples)):
        try:
            value = int(metadata.get(column, ""))
        except ValueError:
            value = None
        if value != expected:
            mismatches.append(column)
    if metadata.get("timer_design") != design:
        mismatches.append("timer_design")
    if metadata.get("timer_contrast") != contrast:
        mismatches.append("timer_contrast")
    if mismatches:
        raise ValueError(
            "R core-runtime identity does not match the selected benchmark input: "
            f"{', '.join(mismatches)}"
        )


def read_r_phases(runtime_rows, runtime_path, cpus, repetition):
    if runtime_rows is None:
        return None, None, None, None, None
    selected = [
        row
        for row in runtime_rows
        if int(row.get("cpus", cpus)) == cpus and int(row.get("repetition", repetition)) == repetition
    ]
    if not selected:
        raise ValueError(f"R reference runtime has no {cpus}-CPU repetition {repetition}: {runtime_path}")
    phases = {row["step"]: float(row["elapsed_seconds"]) for row in selected}
    core_seconds = phases.get("deseq_and_results")
    if core_seconds is None:
        raise ValueError(f"R reference runtime has no deseq_and_results timing for {cpus}-CPU repetition {repetition}")
    vst_seconds = phases.get("vst")
    core_source = "standalone_r_reference_runtime_tsv:deseq_and_results" if core_seconds is not None else None
    vst_source = "standalone_r_reference_runtime_tsv:vst" if vst_seconds is not None else None
    return core_seconds, vst_seconds, core_source, vst_source, runtime_path


def create_run_root(path):
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to reuse pre-existing benchmark run root: {path}") from exc


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combined_sha256(paths, root):
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.as_posix()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest() if paths else None


def portable_path(path, root):
    if path is None:
        return None
    return os.path.relpath(Path(path).resolve(), root)


def git_provenance(path):
    root_result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if root_result.returncode != 0:
        raise ValueError(f"Could not locate the git repository containing {path}")
    root = Path(root_result.stdout.strip()).resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout
    return root, commit, bool(status.strip())


def referenced_module_files(workflow):
    include_paths = re.findall(r"\bfrom\s+['\"]([^'\"]+)['\"]", workflow.read_text(encoding="utf-8"))
    files = set()
    for include_path in include_paths:
        main_path = (workflow.parent / include_path).resolve()
        if not main_path.exists() and main_path.with_suffix(".nf").exists():
            main_path = main_path.with_suffix(".nf")
        if not main_path.exists():
            raise ValueError(f"Included module does not exist: {main_path}")
        files.add(main_path)
        for name in ("meta.yml", "environment.yml", "environment.yaml"):
            candidate = main_path.parent / name
            if candidate.exists():
                files.add(candidate)
        templates = main_path.parent / "templates"
        if templates.exists():
            files.update(
                path
                for path in templates.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
            )
    return sorted(files, key=lambda value: value.as_posix())


def find_environment_yaml(explicit_path, workflow):
    if explicit_path is not None:
        resolved = explicit_path.resolve()
        if not resolved.is_file():
            raise ValueError(f"Environment YAML does not exist: {resolved}")
        return resolved
    candidates = [
        workflow.parent / "environment.yml",
        workflow.parent / "environment.yaml",
        workflow.parent.parent / "environment.yml",
        workflow.parent.parent / "environment.yaml",
    ]
    return next((path.resolve() for path in candidates if path.is_file()), None)


def host_memory_bytes():
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def host_hardware():
    processor = platform.processor().strip()
    model = None
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["/usr/sbin/system_profiler", "SPHardwareDataType", "-detailLevel", "mini"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        chip_match = re.search(r"^\s*Chip:\s*(.+)$", result.stdout, re.MULTILINE)
        model_match = re.search(r"^\s*Model Identifier:\s*(.+)$", result.stdout, re.MULTILINE)
        processor = chip_match.group(1).strip() if chip_match else processor
        model = model_match.group(1).strip() if model_match else None
    elif platform.system() == "Linux" and Path("/proc/cpuinfo").is_file():
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        model_match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
        processor = model_match.group(1).strip() if model_match else processor
    return processor or platform.machine(), model


def output_metrics(output_dir, implementation):
    suffix = "*.deseq2.results.tsv" if implementation == "r" else "*.pydeseq2.results.tsv"
    result_files = list(output_dir.glob(f"**/{suffix}"))
    result_checksum = sha256(result_files[0]) if len(result_files) == 1 else None
    data_patterns = (
        "*.deseq2.results.tsv",
        "*.pydeseq2.results.tsv",
        "*.normalised_counts.tsv",
        "*.vst.tsv",
        "*.deseq2.sizefactors.tsv",
        "*.pydeseq2.sizefactors.tsv",
        "*.pydeseq2.normalization_factors.tsv",
    )
    data_files = {path for pattern in data_patterns for path in output_dir.glob(f"**/{pattern}")}
    data_checksum = combined_sha256(data_files, output_dir)
    output_bytes = sum(path.stat().st_size for path in output_dir.glob("**/*") if path.is_file())
    return result_checksum, data_checksum, output_bytes


def read_versions(output_dir):
    python_sessions = list(output_dir.glob("**/*.Python_sessionInfo.log"))
    if len(python_sessions) == 1:
        payload = read_json_payload(python_sessions[0])
        return {
            "r_version": None,
            "deseq2_version": None,
            "python_version": payload.get("python"),
            "pydeseq2_version": payload.get("pydeseq2"),
        }

    r_sessions = list(output_dir.glob("**/*.R_sessionInfo.log"))
    if len(r_sessions) != 1:
        raise ValueError(f"Expected one R or Python session record beneath {output_dir}")
    text = r_sessions[0].read_text(encoding="utf-8")
    r_match = re.search(r"^R version\s+(\S+)", text, re.MULTILINE)
    deseq2_match = re.search(r"\bDESeq2_([0-9][A-Za-z0-9_.-]*)", text)
    if not r_match or not deseq2_match:
        raise ValueError(f"Could not read R and DESeq2 versions from {r_sessions[0]}")
    return {
        "r_version": r_match.group(1),
        "deseq2_version": deseq2_match.group(1),
        "python_version": None,
        "pydeseq2_version": None,
    }


def read_json_payload(path):
    text = path.read_text(encoding="utf-8")
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON payload found in {path}")
    return json.loads(text[start:])


def verify_cpu_evidence(output_dir, work_dir, implementation, requested_cpus):
    if implementation == "pydeseq2":
        session_files = list(output_dir.glob("**/*.Python_sessionInfo.log"))
        if len(session_files) != 1:
            raise ValueError(f"Expected one PyDESeq2 session record beneath {output_dir}")
        observed = int(read_json_payload(session_files[0]).get("options", {}).get("cores"))
        evidence = session_files
    else:
        worker_counts = set()
        evidence_paths = []
        for log_path in work_dir.glob("**/.command.log"):
            matches = re.findall(
                r"dispersion estimates(?:, fitting model and testing)?:\s+(\d+) workers",
                log_path.read_text(encoding="utf-8"),
            )
            if matches:
                worker_counts.update(int(value) for value in matches)
                evidence_paths.append(log_path)
        if requested_cpus not in worker_counts:
            raise ValueError(f"R task logs do not prove {requested_cpus} workers beneath {work_dir}")
        observed = requested_cpus
        evidence = evidence_paths
    if observed != requested_cpus:
        raise ValueError(f"Requested {requested_cpus} CPUs but observed {observed} for {implementation}")
    return observed, evidence


def verify_native_math_threads(work_dir):
    required = (
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        "VECLIB_MAXIMUM_THREADS=1",
    )
    wrappers = list(work_dir.glob("**/.command.run"))
    matching = [path for path in wrappers if all(value in path.read_text(encoding="utf-8") for value in required)]
    if len(matching) != 1:
        raise ValueError(f"Expected one task wrapper with one native math thread per worker beneath {work_dir}")
    return 1, matching[0]


def nextflow_version(executable, env):
    result = subprocess.run(
        [executable, "-version"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    match = re.search(r"version\s+([0-9][0-9.]*)", result.stdout, re.IGNORECASE)
    return match.group(1) if match else None


def dataset_identity(input_dir, dataset):
    count_files = list(input_dir.glob(f"{dataset.lower()}_counts.tsv"))
    if len(count_files) != 1:
        raise ValueError(f"Expected one {dataset.lower()}_counts.tsv in {input_dir}")
    with count_files[0].open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        genes = sum(1 for _ in handle)
    summary_files = list(input_dir.glob(f"{dataset.lower()}_deseq2_summary.tsv"))
    if len(summary_files) != 1:
        raise ValueError(f"Expected one {dataset.lower()}_deseq2_summary.tsv in {input_dir}")
    with summary_files[0].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 1 or not rows[0].get("design", "").strip() or not rows[0].get("contrast", "").strip():
        raise ValueError(f"Expected one summary row with design and contrast in {summary_files[0]}")
    return genes, len(header) - 1, rows[0]["design"].strip(), rows[0]["contrast"].strip()


def write_timings(path, rows):
    columns = [
        "dataset",
        "implementation",
        "cpus",
        "repetition",
        "implementation_order",
        "execution_sequence",
        "genes",
        "samples",
        "design",
        "contrast",
        "task_seconds",
        "wall_seconds",
        "workflow_seconds",
        "core_seconds",
        "vst_seconds",
        "peak_rss_bytes",
        "output_bytes",
        "task_timing_source",
        "workflow_timing_source",
        "timing_source",
        "core_timing_source",
        "core_timing_file",
        "core_timer_runtime",
        "core_timer_runtime_version",
        "core_timer_engine",
        "core_timer_engine_version",
        "core_timer_host_system",
        "core_timer_host_machine",
        "core_timer_dataset",
        "core_timer_genes",
        "core_timer_samples",
        "core_timer_design",
        "core_timer_contrast",
        "vst_timing_source",
        "vst_timing_file",
        "result_sha256",
        "data_outputs_sha256",
        "observed_cpus",
        "cpu_evidence",
        "native_math_threads_per_worker",
        "thread_evidence",
        "profile",
        "r_version",
        "deseq2_version",
        "python_version",
        "pydeseq2_version",
        "nextflow_version",
        "host_system",
        "host_release",
        "host_machine",
        "host_processor",
        "host_model",
        "host_platform",
        "host_logical_cpus",
        "host_memory_bytes",
        "git_commit",
        "git_dirty",
        "started_at_utc",
        "completed_at_utc",
        "command",
        "command_cwd",
        "workflow_path",
        "workflow_sha256",
        "config_path",
        "config_sha256",
        "input_dir",
        "input_sha256",
        "module_files",
        "module_sha256",
        "benchmark_script_path",
        "benchmark_script_sha256",
        "environment_yaml_path",
        "environment_yaml_sha256",
        "r_core_runtime_path",
        "nxf_home",
        "output_dir",
        "work_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least 1")
    cpus_values = [int(value.strip()) for value in args.cpus.split(",") if value.strip()]
    if not cpus_values or any(value < 1 for value in cpus_values) or len(cpus_values) != len(set(cpus_values)):
        raise ValueError("--cpus must contain unique positive integers")
    implementations = [value.strip() for value in args.implementations.split(",") if value.strip()]
    if not set(implementations).issubset({"r", "pydeseq2"}):
        raise ValueError("--implementations supports only r and pydeseq2")
    if not implementations or len(implementations) != len(set(implementations)):
        raise ValueError("--implementations must contain unique values")

    workflow = args.workflow.resolve()
    config = args.config.resolve()
    input_dir = args.input_dir.resolve()
    benchmark_root = args.outdir.resolve()
    r_core_runtime = args.r_core_runtime.resolve() if args.r_core_runtime else None
    r_env_bin = args.r_env_bin.resolve() if args.r_env_bin else None
    pydeseq2_env_bin = args.pydeseq2_env_bin.resolve() if args.pydeseq2_env_bin else None
    validate_external_r_runtime_setup(
        args.profile,
        r_core_runtime,
        r_env_bin,
        pydeseq2_env_bin,
    )
    genes, samples, design, contrast = dataset_identity(input_dir, args.dataset)
    timer_identity = {
        "timer_dataset": args.dataset,
        "timer_genes": genes,
        "timer_samples": samples,
        "timer_design": design,
        "timer_contrast": contrast,
    }
    r_runtime_rows, r_timer_metadata = load_r_runtime(r_core_runtime) if r_core_runtime else (None, None)
    if r_timer_metadata:
        validate_r_timer_identity(r_timer_metadata, args.dataset, genes, samples, design, contrast)
    if r_timer_metadata and (
        r_timer_metadata["timer_host_system"] != platform.system()
        or r_timer_metadata["timer_host_machine"] != platform.machine()
    ):
        raise ValueError("R core-runtime timer host does not match the current host")
    nxf_home = args.nxf_home.resolve() if args.nxf_home else None
    environment_yaml = find_environment_yaml(args.environment_yaml, workflow)
    timings_path = benchmark_root / f"{args.dataset.lower()}_benchmark_timings.tsv"
    env = os.environ.copy()
    if nxf_home:
        env["NXF_HOME"] = str(nxf_home)
    workflow_nextflow_version = nextflow_version(args.nextflow, env)
    git_root, git_commit, git_dirty = git_provenance(workflow.parent)
    module_files = referenced_module_files(workflow)
    benchmark_script = Path(__file__).resolve()
    input_files = [path for path in input_dir.rglob("*") if path.is_file()]
    provenance = {
        "workflow_path": portable_path(workflow, benchmark_root),
        "workflow_sha256": sha256(workflow),
        "config_path": portable_path(config, benchmark_root),
        "config_sha256": sha256(config),
        "input_dir": portable_path(input_dir, benchmark_root),
        "input_sha256": combined_sha256(input_files, input_dir),
        "module_files": json.dumps(
            [portable_path(path, benchmark_root) for path in module_files],
            separators=(",", ":"),
        ),
        "module_sha256": combined_sha256(module_files, git_root),
        "benchmark_script_path": portable_path(benchmark_script, benchmark_root),
        "benchmark_script_sha256": sha256(benchmark_script),
        "environment_yaml_path": portable_path(environment_yaml, benchmark_root),
        "environment_yaml_sha256": sha256(environment_yaml) if environment_yaml else None,
        "r_core_runtime_path": portable_path(r_core_runtime, benchmark_root),
        "nxf_home": portable_path(nxf_home, benchmark_root),
    }
    host_processor, host_model = host_hardware()
    host = {
        "host_system": platform.system(),
        "host_release": platform.release(),
        "host_machine": platform.machine(),
        "host_processor": host_processor,
        "host_model": host_model,
        "host_platform": platform.platform(),
        "host_logical_cpus": os.cpu_count(),
        "host_memory_bytes": host_memory_bytes(),
    }

    rows = []
    execution_sequence = 0
    for cpu_index, cpus in enumerate(cpus_values):
        for repetition in range(0, args.repetitions + 1):
            offset = (cpu_index + repetition) % len(implementations)
            implementation_order = implementations[offset:] + implementations[:offset]
            for order_position, implementation in enumerate(implementation_order, start=1):
                execution_sequence += 1
                run_label = "warmup" if repetition == 0 else f"run_{repetition}"
                run_root = benchmark_root / f"{implementation}_{cpus}cpu" / run_label
                output_dir = run_root / "results"
                work_dir = run_root / "work"
                trace_path = run_root / "trace.tsv"
                log_path = run_root / "nextflow.log"
                create_run_root(run_root)

                command = [
                    args.nextflow,
                    "-C",
                    str(config),
                    "run",
                    workflow.name,
                    "--input_dir",
                    str(input_dir),
                    "--outdir",
                    str(output_dir),
                    "--cpus",
                    str(cpus),
                    "--implementation",
                    implementation,
                    "-profile",
                    args.profile,
                    "-with-trace",
                    str(trace_path),
                    "-work-dir",
                    str(work_dir),
                ]
                if r_env_bin:
                    command.extend(["--r_env_bin", str(r_env_bin)])
                if pydeseq2_env_bin:
                    command.extend(["--pydeseq2_env_bin", str(pydeseq2_env_bin)])
                started_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                started = time.perf_counter()
                result = subprocess.run(
                    command,
                    cwd=workflow.parent,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                workflow_seconds = time.perf_counter() - started
                completed_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                log_path.write_text(result.stdout, encoding="utf-8")
                if result.returncode != 0:
                    raise RuntimeError(f"Benchmark run failed; inspect {log_path}")
                observed_cpus, cpu_evidence = verify_cpu_evidence(
                    output_dir,
                    work_dir,
                    implementation,
                    cpus,
                )
                native_math_threads, thread_evidence = verify_native_math_threads(work_dir)
                versions = read_versions(output_dir)
                if implementation == "r" and r_timer_metadata:
                    validate_r_timer_metadata(
                        r_timer_metadata,
                        versions,
                        host["host_system"],
                        host["host_machine"],
                    )
                if repetition == 0:
                    continue

                wall_seconds, peak_rss = read_nextflow_metrics(trace_path, implementation)
                task_timing_source = "nextflow_trace"
                if wall_seconds is None:
                    wall_seconds = read_command_trace(work_dir)
                    task_timing_source = "command_trace"
                if wall_seconds is None:
                    raise ValueError(f"Could not recover task runtime from trace files beneath {run_root}")

                if implementation == "pydeseq2":
                    core_seconds, vst_seconds, core_source, vst_source, phase_file = read_pydeseq2_phases(output_dir)
                    timer_metadata = {
                        "timer_runtime": "Python",
                        "timer_runtime_version": versions["python_version"],
                        "timer_engine": "PyDESeq2",
                        "timer_engine_version": versions["pydeseq2_version"],
                        "timer_host_system": host["host_system"] if args.profile == "local" else None,
                        "timer_host_machine": host["host_machine"] if args.profile == "local" else None,
                        **timer_identity,
                    }
                else:
                    core_seconds, vst_seconds, core_source, vst_source, phase_file = read_r_phases(
                        r_runtime_rows,
                        r_core_runtime,
                        cpus,
                        repetition,
                    )
                    timer_metadata = r_timer_metadata or {column: None for column in TIMER_PROVENANCE_COLUMNS}
                result_checksum, data_checksum, output_bytes = output_metrics(output_dir, implementation)
                rows.append(
                    {
                        "dataset": args.dataset,
                        "implementation": implementation,
                        "cpus": cpus,
                        "repetition": repetition,
                        "implementation_order": order_position,
                          "execution_sequence": execution_sequence,
                          "genes": genes,
                          "samples": samples,
                          "design": design,
                          "contrast": contrast,
                        "task_seconds": wall_seconds,
                        "wall_seconds": wall_seconds,
                        "workflow_seconds": workflow_seconds,
                        "core_seconds": core_seconds,
                        "vst_seconds": vst_seconds,
                        "peak_rss_bytes": peak_rss,
                        "output_bytes": output_bytes,
                        "task_timing_source": task_timing_source,
                        "workflow_timing_source": "python_perf_counter",
                        "timing_source": task_timing_source,
                        "core_timing_source": core_source,
                        "core_timing_file": portable_path(phase_file, benchmark_root),
                        **{f"core_{column}": value for column, value in timer_metadata.items()},
                        "vst_timing_source": vst_source,
                        "vst_timing_file": portable_path(phase_file, benchmark_root),
                        "result_sha256": result_checksum,
                        "data_outputs_sha256": data_checksum,
                        "observed_cpus": observed_cpus,
                        "cpu_evidence": json.dumps(
                            [portable_path(path, benchmark_root) for path in cpu_evidence],
                            separators=(",", ":"),
                        ),
                        "native_math_threads_per_worker": native_math_threads,
                        "thread_evidence": portable_path(thread_evidence, benchmark_root),
                        "profile": args.profile,
                        **versions,
                        "nextflow_version": workflow_nextflow_version,
                        **host,
                        "git_commit": git_commit,
                        "git_dirty": git_dirty,
                        "started_at_utc": started_at_utc,
                        "completed_at_utc": completed_at_utc,
                        "command": shlex.join(command),
                        "command_cwd": portable_path(workflow.parent, benchmark_root),
                        **provenance,
                        "output_dir": portable_path(output_dir, benchmark_root),
                        "work_dir": portable_path(work_dir, benchmark_root),
                    }
                )
                write_timings(timings_path, rows)

    print(timings_path)


if __name__ == "__main__":
    main()
