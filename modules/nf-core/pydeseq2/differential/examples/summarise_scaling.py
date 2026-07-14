#!/usr/bin/env python3

import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd


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
    parser = argparse.ArgumentParser(description="Summarise Pasilla and Pickrell DESeq2 scaling runs.")
    parser.add_argument("--timings", required=True, nargs="+", help="Per-run timing TSV files.")
    parser.add_argument("--outdir", required=True, type=Path, help="Summary output directory.")
    parser.add_argument(
        "--r-core-runtime",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Matched local standalone R core-runtime TSV override; repeat once per dataset.",
    )
    return parser.parse_args()


def deterministic(values):
    return bool(not values.empty and values.notna().all() and values.nunique(dropna=False) == 1)


def interquartile_range(values):
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return numeric.quantile(0.75) - numeric.quantile(0.25) if not numeric.empty else float("nan")


def median_absolute_deviation(values):
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return (numeric - numeric.median()).abs().median() if not numeric.empty else float("nan")


def jsonable(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def load_timings(paths):
    tables = []
    for value in paths:
        path = Path(value).resolve()
        table = pd.read_csv(path, sep="\t")
        table["_timing_table_dir"] = str(path.parent)
        table["_timing_table_path"] = str(path)
        tables.append(table)
    timings = pd.concat(tables, ignore_index=True)

    # Preserve input compatibility while keeping task and workflow clocks explicit.
    if "task_seconds" not in timings.columns and "wall_seconds" in timings.columns:
        timings["task_seconds"] = timings["wall_seconds"]
    if "task_timing_source" not in timings.columns and "timing_source" in timings.columns:
        timings["task_timing_source"] = timings["timing_source"]
    if "workflow_timing_source" not in timings.columns and "workflow_seconds" in timings.columns:
        timings["workflow_timing_source"] = "legacy_python_perf_counter"
    for column in ("core_timing_source", "core_timing_file", "vst_timing_source", "vst_timing_file"):
        if column not in timings.columns:
            timings[column] = pd.NA
    for column in TIMER_PROVENANCE_COLUMNS:
        benchmark_column = f"core_{column}"
        if benchmark_column not in timings.columns:
            timings[benchmark_column] = pd.NA

    # Legacy local tables predate explicit core-timer columns, but their PyDESeq2
    # core timer is the module task itself, so its recorded session and host are authoritative.
    py_mask = timings["implementation"].eq("pydeseq2") & timings["core_seconds"].notna()
    timings.loc[py_mask, "core_timer_runtime"] = timings.loc[
        py_mask, "core_timer_runtime"
    ].fillna("Python")
    runtime_versions = timings.loc[py_mask, "core_timer_runtime_version"]
    timings.loc[py_mask, "core_timer_runtime_version"] = runtime_versions.where(
        runtime_versions.notna(), timings.loc[py_mask, "python_version"]
    )
    timings.loc[py_mask, "core_timer_engine"] = timings.loc[py_mask, "core_timer_engine"].fillna(
        "PyDESeq2"
    )
    engine_versions = timings.loc[py_mask, "core_timer_engine_version"]
    timings.loc[py_mask, "core_timer_engine_version"] = engine_versions.where(
        engine_versions.notna(), timings.loc[py_mask, "pydeseq2_version"]
    )
    local_py_mask = py_mask & timings["profile"].eq("local")
    timings.loc[local_py_mask, "core_timer_host_system"] = timings.loc[
        local_py_mask, "core_timer_host_system"
    ].fillna(timings.loc[local_py_mask, "host_system"])
    timings.loc[local_py_mask, "core_timer_host_machine"] = timings.loc[
        local_py_mask, "core_timer_host_machine"
    ].fillna(timings.loc[local_py_mask, "host_machine"])
    for timer_column, timing_column in (
        ("timer_dataset", "dataset"),
        ("timer_genes", "genes"),
        ("timer_samples", "samples"),
        ("timer_design", "design"),
        ("timer_contrast", "contrast"),
    ):
        if timing_column in timings.columns:
            benchmark_column = f"core_{timer_column}"
            timings.loc[py_mask, benchmark_column] = timings.loc[py_mask, benchmark_column].where(
                timings.loc[py_mask, benchmark_column].notna(),
                timings.loc[py_mask, timing_column],
            )
    return timings


def parse_r_core_runtime_specs(values):
    overrides = {}
    seen_datasets = set()
    for value in values:
        dataset, separator, raw_path = value.partition("=")
        dataset = dataset.strip()
        raw_path = raw_path.strip()
        if not separator or not dataset or not raw_path:
            raise ValueError("--r-core-runtime must use DATASET=PATH")
        dataset_key = dataset.casefold()
        if dataset_key in seen_datasets:
            raise ValueError(f"Duplicate --r-core-runtime dataset: {dataset}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"R core-runtime TSV does not exist: {path}")
        seen_datasets.add(dataset_key)
        overrides[dataset] = path
    return overrides


def apply_r_core_runtime_overrides(timings, overrides):
    effective = timings.copy()
    provenance_columns = (
        "core_timing_source",
        "core_timing_file",
        *(f"core_{column}" for column in TIMER_PROVENANCE_COLUMNS),
        "vst_timing_source",
        "vst_timing_file",
        "r_core_runtime_path",
    )
    for column in provenance_columns:
        if column not in effective.columns:
            effective[column] = pd.NA

    for dataset, runtime_path in overrides.items():
        runtime_path = Path(runtime_path).resolve()
        r_dataset_names = effective.loc[
            effective["implementation"].eq("r")
            & effective["dataset"].astype(str).str.casefold().eq(dataset.casefold()),
            "dataset",
        ].astype(str).unique()
        if len(r_dataset_names) == 0:
            raise ValueError(f"No R timing cells found for dataset {dataset}")
        if len(r_dataset_names) > 1:
            raise ValueError(f"Ambiguous case-insensitive R timing datasets for {dataset}")
        r_mask = effective["dataset"].astype(str).eq(r_dataset_names[0]) & effective[
            "implementation"
        ].eq("r")
        profile_values = effective.loc[r_mask, "profile"]
        profiles = profile_values.dropna().astype(str).str.strip().unique()
        if (
            profile_values.isna().any()
            or profile_values.astype(str).str.strip().eq("").any()
            or len(profiles) != 1
            or profiles[0] != "local"
        ):
            raise ValueError(f"R core-runtime overrides require a matched local benchmark for {dataset}")
        if "_timing_table_dir" in effective.columns:
            provenance_paths = effective.loc[r_mask, "_timing_table_dir"].map(
                lambda directory: os.path.relpath(runtime_path, directory)
            )
        else:
            provenance_paths = pd.Series(str(runtime_path), index=effective.index[r_mask])

        runtime = pd.read_csv(runtime_path, sep="\t")
        required = {"cpus", "repetition", "step", "elapsed_seconds", *TIMER_PROVENANCE_COLUMNS}
        missing = required.difference(runtime.columns)
        if missing:
            raise ValueError(
                f"R core-runtime TSV for {dataset} is missing columns: {', '.join(sorted(missing))}"
            )
        runtime = runtime.copy()
        timer_metadata = {}
        for column in TIMER_PROVENANCE_COLUMNS:
            if runtime[column].isna().any():
                raise ValueError(f"R core-runtime {column} must be non-empty and constant for {dataset}")
            values = runtime[column].astype(str).str.strip().unique()
            if len(values) != 1 or not values[0]:
                raise ValueError(f"R core-runtime {column} must be non-empty and constant for {dataset}")
            timer_metadata[column] = values[0]
        if timer_metadata["timer_runtime"] != "R" or timer_metadata["timer_engine"] != "DESeq2":
            raise ValueError(f"R core-runtime timer runtime/engine must be R and DESeq2 for {dataset}")

        expected_metadata = {
            "timer_runtime_version": "r_version",
            "timer_engine_version": "deseq2_version",
            "timer_host_system": "host_system",
            "timer_host_machine": "host_machine",
        }
        for timer_column, timing_column in expected_metadata.items():
            values = effective.loc[r_mask, timing_column].dropna().astype(str).str.strip().unique()
            if len(values) != 1 or timer_metadata[timer_column] != values[0]:
                raise ValueError(
                    f"R core-runtime {timer_column} does not match R timing metadata for {dataset}"
                )
        if timer_metadata["timer_dataset"].casefold() != r_dataset_names[0].casefold():
            raise ValueError(f"R core-runtime timer_dataset does not match R timing metadata for {dataset}")
        missing_identity = {"genes", "samples", "design", "contrast"}.difference(effective.columns)
        if missing_identity:
            raise ValueError(
                f"R timing metadata for {dataset} is missing identity columns: "
                f"{', '.join(sorted(missing_identity))}"
            )
        for timer_column, timing_column in (("timer_genes", "genes"), ("timer_samples", "samples")):
            timer_value = pd.to_numeric(pd.Series([timer_metadata[timer_column]]), errors="coerce").iloc[0]
            timing_values = pd.to_numeric(effective.loc[r_mask, timing_column], errors="coerce").dropna().unique()
            if (
                pd.isna(timer_value)
                or not float(timer_value).is_integer()
                or timer_value < 1
                or len(timing_values) != 1
                or timer_value != timing_values[0]
            ):
                raise ValueError(
                    f"R core-runtime {timer_column} does not match R timing metadata for {dataset}"
                )
        for timer_column, timing_column in (("timer_design", "design"), ("timer_contrast", "contrast")):
            timing_values = effective.loc[r_mask, timing_column].dropna().astype(str).str.strip().unique()
            if len(timing_values) != 1 or not timing_values[0] or timer_metadata[timer_column] != timing_values[0]:
                raise ValueError(
                    f"R core-runtime {timer_column} does not match R timing metadata for {dataset}"
                )
        for column in ("cpus", "repetition"):
            numeric = pd.to_numeric(runtime[column], errors="coerce")
            if numeric.isna().any() or not numeric.mod(1).eq(0).all() or (numeric < 1).any():
                raise ValueError(f"R core-runtime {column} values must be positive integers for {dataset}")
            runtime[column] = numeric.astype(int)
        if runtime["step"].isna().any() or runtime["step"].astype(str).str.strip().eq("").any():
            raise ValueError(f"R core-runtime step values must not be empty for {dataset}")
        runtime["step"] = runtime["step"].astype(str)
        runtime["elapsed_seconds"] = pd.to_numeric(runtime["elapsed_seconds"], errors="coerce")
        finite = runtime["elapsed_seconds"].notna() & runtime["elapsed_seconds"].map(math.isfinite)
        if not finite.all() or (runtime["elapsed_seconds"] < 0).any():
            raise ValueError(f"R core-runtime elapsed_seconds must be finite and non-negative for {dataset}")

        runtime_keys = ["cpus", "repetition", "step"]
        if runtime.duplicated(runtime_keys, keep=False).any():
            raise ValueError(f"R core-runtime TSV contains duplicate (cpus, repetition, step) rows for {dataset}")

        r_keys = effective.loc[r_mask, ["cpus", "repetition"]].copy()
        for column in ("cpus", "repetition"):
            numeric = pd.to_numeric(r_keys[column], errors="coerce")
            if numeric.isna().any() or not numeric.mod(1).eq(0).all():
                raise ValueError(f"R timing-cell {column} values must be integers for {dataset}")
            r_keys[column] = numeric.astype(int)
        expected_keys = pd.MultiIndex.from_frame(r_keys, names=["cpus", "repetition"])

        core = runtime.loc[runtime["step"] == "deseq_and_results"]
        core_keys = pd.MultiIndex.from_frame(core[["cpus", "repetition"]])
        if not expected_keys.difference(core_keys).empty or not core_keys.difference(expected_keys).empty:
            raise ValueError(
                f"R core-runtime deseq_and_results rows do not exactly match R timing cells for {dataset}"
            )
        core_values = core.set_index(["cpus", "repetition"])["elapsed_seconds"].reindex(expected_keys)

        vst = runtime.loc[runtime["step"] == "vst"]
        if vst.empty:
            vst_values = pd.Series(float("nan"), index=expected_keys)
            vst_source = pd.NA
            vst_file = pd.NA
        else:
            vst_keys = pd.MultiIndex.from_frame(vst[["cpus", "repetition"]])
            if not expected_keys.difference(vst_keys).empty or not vst_keys.difference(expected_keys).empty:
                raise ValueError(f"R core-runtime vst rows do not exactly match R timing cells for {dataset}")
            vst_values = vst.set_index(["cpus", "repetition"])["elapsed_seconds"].reindex(expected_keys)
            vst_source = "standalone_r_reference_runtime_tsv:vst"
            vst_file = provenance_paths.to_numpy()

        effective.loc[r_mask, "core_seconds"] = core_values.to_numpy()
        effective.loc[r_mask, "vst_seconds"] = vst_values.to_numpy()
        effective.loc[r_mask, "core_timing_source"] = (
            "standalone_r_reference_runtime_tsv:deseq_and_results"
        )
        effective.loc[r_mask, "core_timing_file"] = provenance_paths.to_numpy()
        for column, value in timer_metadata.items():
            effective.loc[r_mask, f"core_{column}"] = value
        effective.loc[r_mask, "vst_timing_source"] = vst_source
        effective.loc[r_mask, "vst_timing_file"] = vst_file
        effective.loc[r_mask, "r_core_runtime_path"] = provenance_paths.to_numpy()

    return effective


def validate_complete_matrix(timings):
    required = {
        "dataset",
        "implementation",
        "cpus",
        "repetition",
        "genes",
        "samples",
        "task_seconds",
        "workflow_seconds",
        "task_timing_source",
        "workflow_timing_source",
        "core_seconds",
        "vst_seconds",
        "peak_rss_bytes",
        "output_bytes",
        "result_sha256",
        "data_outputs_sha256",
        "observed_cpus",
        "native_math_threads_per_worker",
        "profile",
        "r_version",
        "deseq2_version",
        "python_version",
        "pydeseq2_version",
        "nextflow_version",
        "host_system",
        "host_machine",
        "output_dir",
    }
    missing_columns = required.difference(timings.columns)
    if missing_columns:
        raise ValueError(f"Timing tables are missing columns: {', '.join(sorted(missing_columns))}")

    key_columns = ["dataset", "implementation", "cpus", "repetition"]
    if timings[key_columns].isna().any().any():
        raise ValueError("Dataset, implementation, CPU, and repetition keys must not be empty")
    timings["dataset"] = timings["dataset"].astype(str)
    timings["implementation"] = timings["implementation"].astype(str)
    for column in ("cpus", "repetition", "genes", "samples", "observed_cpus", "native_math_threads_per_worker"):
        numeric = pd.to_numeric(timings[column], errors="coerce")
        if numeric.isna().any() or not numeric.mod(1).eq(0).all():
            raise ValueError(f"{column} must contain integers in every timing row")
        timings[column] = numeric.astype(int)

    implementations = set(timings["implementation"])
    if implementations != {"r", "pydeseq2"}:
        raise ValueError("Scaling comparisons require exactly the r and pydeseq2 implementations")
    if (timings["cpus"] < 1).any() or (timings["repetition"] < 1).any():
        raise ValueError("Measured timing rows require positive CPU counts and repetitions starting at 1")
    if timings.duplicated(key_columns, keep=False).any():
        duplicates = timings.loc[timings.duplicated(key_columns, keep=False), key_columns]
        raise ValueError(f"Duplicate benchmark cells:\n{duplicates.to_string(index=False)}")

    repetitions = sorted(int(value) for value in timings["repetition"].unique())
    expected_repetitions = list(range(1, max(repetitions) + 1))
    if repetitions != expected_repetitions:
        raise ValueError(f"Measured repetitions must be contiguous from 1; observed {repetitions}")
    datasets = sorted(timings["dataset"].unique())
    cpus = sorted(int(value) for value in timings["cpus"].unique())
    expected = pd.MultiIndex.from_product(
        [datasets, ["r", "pydeseq2"], cpus, expected_repetitions],
        names=key_columns,
    )
    observed = pd.MultiIndex.from_frame(timings[key_columns])
    missing_cells = expected.difference(observed)
    if not missing_cells.empty:
        preview = missing_cells.to_frame(index=False).head(20)
        raise ValueError(f"Incomplete benchmark matrix; missing cells include:\n{preview.to_string(index=False)}")

    dimensions = timings.groupby("dataset")[["genes", "samples"]].nunique(dropna=False)
    if dimensions.gt(1).any(axis=None):
        raise ValueError("Gene and sample dimensions must remain constant within each dataset")

    for column in ("task_seconds", "workflow_seconds"):
        timings[column] = pd.to_numeric(timings[column], errors="coerce")
        finite = timings[column].notna() & timings[column].map(math.isfinite)
        if not finite.all() or (timings[column] <= 0).any():
            raise ValueError(f"{column} must contain a finite positive value for every comparison cell")
    setting_columns = ["dataset", "implementation", "cpus"]
    setting_sizes = timings.groupby(setting_columns).size()
    for column in ("core_seconds", "vst_seconds"):
        timings[column] = pd.to_numeric(timings[column], errors="coerce")
        counts = timings.groupby(setting_columns)[column].count()
        partial = counts.ne(0) & counts.ne(setting_sizes)
        if partial.any():
            settings = partial.loc[partial].index.tolist()
            raise ValueError(f"{column} is present for only part of benchmark settings: {settings}")
    if not timings["observed_cpus"].eq(timings["cpus"]).all():
        raise ValueError("Observed CPU evidence does not match the requested CPU count")
    if not timings["native_math_threads_per_worker"].eq(1).all():
        raise ValueError("Every run must prove one native math thread per worker")

    for column in ("task_timing_source", "workflow_timing_source"):
        if timings[column].isna().any() or timings[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} must be recorded for every run")
        mixed = timings.groupby(setting_columns)[column].nunique(dropna=False).gt(1)
        if mixed.any():
            settings = mixed.loc[mixed].index.tolist()
            raise ValueError(f"{column} changes within benchmark settings: {settings}")

    return datasets, cpus, expected_repetitions


def collect_pydeseq2_phases(timings):
    rows = []
    selected = timings.loc[timings["implementation"] == "pydeseq2"]
    for timing in selected.to_dict(orient="records"):
        output_dir = Path(timing["output_dir"])
        if not output_dir.is_absolute():
            output_dir = Path(timing["_timing_table_dir"]) / output_dir
        runtime_files = list(output_dir.glob("**/*.pydeseq2.runtime.tsv"))
        if len(runtime_files) != 1:
            raise ValueError(f"Expected one PyDESeq2 runtime TSV beneath {output_dir}")
        phases = pd.read_csv(runtime_files[0], sep="\t")
        if not {"step", "elapsed_seconds"}.issubset(phases.columns):
            raise ValueError(f"Unexpected PyDESeq2 runtime structure in {runtime_files[0]}")
        if phases["step"].duplicated().any():
            raise ValueError(f"Duplicate phase names in {runtime_files[0]}")
        phases["elapsed_seconds"] = pd.to_numeric(phases["elapsed_seconds"], errors="coerce")
        finite = phases["elapsed_seconds"].notna() & phases["elapsed_seconds"].map(math.isfinite)
        if not finite.all() or (phases["elapsed_seconds"] < 0).any():
            raise ValueError(f"Invalid phase duration in {runtime_files[0]}")
        if "call_count" not in phases.columns:
            phases["call_count"] = pd.NA
        if "timing_semantics" not in phases.columns:
            phases["timing_semantics"] = "legacy_unspecified"
        phases.insert(0, "repetition", timing["repetition"])
        phases.insert(0, "cpus", timing["cpus"])
        phases.insert(0, "samples", timing["samples"])
        phases.insert(0, "genes", timing["genes"])
        phases.insert(0, "dataset", timing["dataset"])
        phases["runtime_file"] = os.path.relpath(runtime_files[0], timing["_timing_table_dir"])
        rows.append(phases)
    return pd.concat(rows, ignore_index=True)


def faster_implementation(comparison, r_column, pydeseq2_column):
    result = pd.Series("tie", index=comparison.index, dtype="object")
    result.loc[comparison[r_column] < comparison[pydeseq2_column]] = "r"
    result.loc[comparison[pydeseq2_column] < comparison[r_column]] = "pydeseq2"
    return result


def performance_crossover(values):
    winners = set(values)
    return "r" in winners and "pydeseq2" in winners


def format_value(value, significant_digits=4):
    return format(float(value), f".{significant_digits}g")


def main():
    args = parse_args()
    timings = load_timings(args.timings)
    timings = apply_r_core_runtime_overrides(
        timings,
        parse_r_core_runtime_specs(args.r_core_runtime),
    )
    datasets, tested_cpus, expected_repetitions = validate_complete_matrix(timings)

    group_columns = ["dataset", "genes", "samples", "implementation", "cpus"]
    aggregations = {
        "repetitions": ("repetition", "nunique"),
        "task_seconds_median": ("task_seconds", "median"),
        "task_seconds_min": ("task_seconds", "min"),
        "task_seconds_max": ("task_seconds", "max"),
        "task_seconds_iqr": ("task_seconds", interquartile_range),
        "task_seconds_mad": ("task_seconds", median_absolute_deviation),
        "workflow_seconds_median": ("workflow_seconds", "median"),
        "workflow_seconds_min": ("workflow_seconds", "min"),
        "workflow_seconds_max": ("workflow_seconds", "max"),
        "workflow_seconds_iqr": ("workflow_seconds", interquartile_range),
        "workflow_seconds_mad": ("workflow_seconds", median_absolute_deviation),
        "core_seconds_median": ("core_seconds", "median"),
        "core_seconds_n": ("core_seconds", "count"),
        "core_seconds_min": ("core_seconds", "min"),
        "core_seconds_max": ("core_seconds", "max"),
        "core_seconds_iqr": ("core_seconds", interquartile_range),
        "core_seconds_mad": ("core_seconds", median_absolute_deviation),
        "vst_seconds_median": ("vst_seconds", "median"),
        "vst_seconds_n": ("vst_seconds", "count"),
        "vst_seconds_min": ("vst_seconds", "min"),
        "vst_seconds_max": ("vst_seconds", "max"),
        "vst_seconds_iqr": ("vst_seconds", interquartile_range),
        "vst_seconds_mad": ("vst_seconds", median_absolute_deviation),
        "peak_rss_bytes_max": ("peak_rss_bytes", "max"),
        "output_bytes_median": ("output_bytes", "median"),
        "results_deterministic": ("result_sha256", deterministic),
        "data_outputs_deterministic": ("data_outputs_sha256", deterministic),
        "observed_cpus_min": ("observed_cpus", "min"),
        "observed_cpus_max": ("observed_cpus", "max"),
        "native_math_threads_per_worker": ("native_math_threads_per_worker", "max"),
        "task_timing_source": ("task_timing_source", "first"),
        "workflow_timing_source": ("workflow_timing_source", "first"),
        "core_timing_source": ("core_timing_source", "first"),
        "vst_timing_source": ("vst_timing_source", "first"),
    }
    metadata_columns = (
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
        "workflow_sha256",
        "config_sha256",
        "input_sha256",
        "module_sha256",
        "benchmark_script_sha256",
        "environment_yaml_sha256",
        *(f"core_{column}" for column in TIMER_METADATA_COLUMNS),
    )
    aggregations.update({column: (column, "first") for column in metadata_columns if column in timings.columns})
    summary = timings.groupby(group_columns, dropna=False).agg(**aggregations).reset_index()
    # Compatibility aliases are explicitly equal to task duration, never workflow duration.
    summary["wall_seconds_median"] = summary["task_seconds_median"]
    summary["wall_seconds_min"] = summary["task_seconds_min"]
    summary["wall_seconds_max"] = summary["task_seconds_max"]

    comparison_index = ["dataset", "genes", "samples", "cpus"]
    task_comparison = summary.pivot(
        index=comparison_index,
        columns="implementation",
        values="task_seconds_median",
    ).rename(columns={"r": "r_task_seconds_median", "pydeseq2": "pydeseq2_task_seconds_median"})
    workflow_comparison = summary.pivot(
        index=comparison_index,
        columns="implementation",
        values="workflow_seconds_median",
    ).rename(columns={"r": "r_workflow_seconds_median", "pydeseq2": "pydeseq2_workflow_seconds_median"})
    comparison = task_comparison.join(workflow_comparison).reset_index()
    comparison_values = [
        "r_task_seconds_median",
        "pydeseq2_task_seconds_median",
        "r_workflow_seconds_median",
        "pydeseq2_workflow_seconds_median",
    ]
    if comparison[comparison_values].isna().any().any():
        raise ValueError("Task or workflow comparison contains missing implementation medians")
    comparison["pydeseq2_over_r_task_ratio"] = (
        comparison["pydeseq2_task_seconds_median"] / comparison["r_task_seconds_median"]
    )
    comparison["pydeseq2_over_r_workflow_ratio"] = (
        comparison["pydeseq2_workflow_seconds_median"] / comparison["r_workflow_seconds_median"]
    )
    comparison["faster_implementation_task"] = faster_implementation(
        comparison,
        "r_task_seconds_median",
        "pydeseq2_task_seconds_median",
    )
    comparison["faster_implementation_workflow"] = faster_implementation(
        comparison,
        "r_workflow_seconds_median",
        "pydeseq2_workflow_seconds_median",
    )
    for metric in ("task", "workflow"):
        cpu_crossover = comparison.groupby("dataset")[f"faster_implementation_{metric}"].agg(
            performance_crossover
        )
        dataset_crossover = comparison.groupby("cpus")[f"faster_implementation_{metric}"].agg(
            performance_crossover
        )
        comparison[f"performance_crossover_{metric}_within_tested_cpus"] = comparison["dataset"].map(
            cpu_crossover
        )
        comparison[f"performance_crossover_{metric}_across_tested_datasets"] = comparison["cpus"].map(
            dataset_crossover
        )
    comparison["r_wall_seconds_median"] = comparison["r_task_seconds_median"]
    comparison["pydeseq2_wall_seconds_median"] = comparison["pydeseq2_task_seconds_median"]
    comparison["pydeseq2_over_r_wall_ratio"] = comparison["pydeseq2_over_r_task_ratio"]
    comparison["faster_implementation"] = comparison["faster_implementation_task"]
    comparison["performance_crossover"] = comparison["performance_crossover_task_within_tested_cpus"]

    cpu_index = ["dataset", "genes", "samples", "implementation"]
    task_scaling = summary.pivot(index=cpu_index, columns="cpus", values="task_seconds_median")
    workflow_scaling = summary.pivot(index=cpu_index, columns="cpus", values="workflow_seconds_median")
    if not {1, 4}.issubset(task_scaling.columns) or not {1, 4}.issubset(workflow_scaling.columns):
        raise ValueError("Both 1- and 4-CPU timing rows are required for 1-to-4 CPU scaling summaries")
    cpu_scaling = task_scaling[[1, 4]].rename(
        columns={1: "task_seconds_1cpu", 4: "task_seconds_4cpu"}
    ).join(
        workflow_scaling[[1, 4]].rename(
            columns={1: "workflow_seconds_1cpu", 4: "workflow_seconds_4cpu"}
        )
    ).reset_index()
    cpu_scaling["task_speedup_1_to_4"] = cpu_scaling["task_seconds_1cpu"] / cpu_scaling["task_seconds_4cpu"]
    cpu_scaling["task_parallel_efficiency_4cpu"] = cpu_scaling["task_speedup_1_to_4"] / 4
    cpu_scaling["workflow_speedup_1_to_4"] = (
        cpu_scaling["workflow_seconds_1cpu"] / cpu_scaling["workflow_seconds_4cpu"]
    )
    cpu_scaling["workflow_parallel_efficiency_4cpu"] = cpu_scaling["workflow_speedup_1_to_4"] / 4
    cpu_scaling["wall_seconds_1cpu"] = cpu_scaling["task_seconds_1cpu"]
    cpu_scaling["wall_seconds_4cpu"] = cpu_scaling["task_seconds_4cpu"]
    cpu_scaling["speedup_1_to_4"] = cpu_scaling["task_speedup_1_to_4"]
    cpu_scaling["parallel_efficiency_4cpu"] = cpu_scaling["task_parallel_efficiency_4cpu"]

    phase_runs = collect_pydeseq2_phases(timings)
    phase_summary = (
        phase_runs.groupby(["dataset", "genes", "samples", "cpus", "step"], dropna=False)
        .agg(
            repetitions=("repetition", "nunique"),
            elapsed_seconds_median=("elapsed_seconds", "median"),
            elapsed_seconds_min=("elapsed_seconds", "min"),
            elapsed_seconds_max=("elapsed_seconds", "max"),
            elapsed_seconds_iqr=("elapsed_seconds", interquartile_range),
            elapsed_seconds_mad=("elapsed_seconds", median_absolute_deviation),
            call_count_min=("call_count", "min"),
            call_count_max=("call_count", "max"),
            timing_semantics=("timing_semantics", "first"),
        )
        .reset_index()
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    effective_columns = [column for column in timings.columns if not column.startswith("_")]
    timings.loc[:, effective_columns].to_csv(
        args.outdir / "benchmark_effective_timings.tsv",
        sep="\t",
        index=False,
    )
    summary.to_csv(args.outdir / "benchmark_run_summary.tsv", sep="\t", index=False)
    comparison.to_csv(args.outdir / "benchmark_scaling_comparison.tsv", sep="\t", index=False)
    cpu_scaling.to_csv(args.outdir / "benchmark_cpu_scaling.tsv", sep="\t", index=False)
    phase_runs.to_csv(args.outdir / "pydeseq2_phase_runs.tsv", sep="\t", index=False)
    phase_summary.to_csv(args.outdir / "pydeseq2_phase_summary.tsv", sep="\t", index=False)
    payload = {
        "matrix": {
            "datasets": datasets,
            "implementations": ["r", "pydeseq2"],
            "cpus": tested_cpus,
            "repetitions": expected_repetitions,
        },
        "run_summary": [
            {key: jsonable(value) for key, value in row.items()}
            for row in summary.to_dict(orient="records")
        ],
        "comparison": [
            {key: jsonable(value) for key, value in row.items()}
            for row in comparison.to_dict(orient="records")
        ],
        "cpu_scaling": [
            {key: jsonable(value) for key, value in row.items()}
            for row in cpu_scaling.to_dict(orient="records")
        ],
        "pydeseq2_phase_summary": [
            {key: jsonable(value) for key, value in row.items()}
            for row in phase_summary.to_dict(orient="records")
        ],
        "interpretation": (
            "Observed task and workflow scaling at the recorded CPU settings; "
            "no implementation was required to be faster and no result is extrapolated beyond tested settings."
        ),
    }
    (args.outdir / "benchmark_scaling_comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = ["# DESeq2 implementation scaling conclusion", ""]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"- {row.dataset} ({row.genes} genes x {row.samples} samples), {row.cpus} CPU: "
            f"task runtime R {format_value(row.r_task_seconds_median)} s vs PyDESeq2 "
            f"{format_value(row.pydeseq2_task_seconds_median)} s "
            f"(Py/R {format_value(row.pydeseq2_over_r_task_ratio, 3)}; "
            f"faster: {row.faster_implementation_task}); workflow runtime R "
            f"{format_value(row.r_workflow_seconds_median)} s vs PyDESeq2 "
            f"{format_value(row.pydeseq2_workflow_seconds_median)} s "
            f"(Py/R {format_value(row.pydeseq2_over_r_workflow_ratio, 3)}; "
            f"faster: {row.faster_implementation_workflow})."
        )
    lines.extend(["", "## CPU scaling", ""])
    for row in cpu_scaling.itertuples(index=False):
        lines.append(
            f"- {row.dataset} {row.implementation}: task 1-to-4 CPU speedup "
            f"{format_value(row.task_speedup_1_to_4, 3)} "
            f"(efficiency {format_value(row.task_parallel_efficiency_4cpu, 3)}); workflow speedup "
            f"{format_value(row.workflow_speedup_1_to_4, 3)} "
            f"(efficiency {format_value(row.workflow_parallel_efficiency_4cpu, 3)})."
        )

    cpu_text = ", ".join(str(value) for value in tested_cpus)
    task_crossovers = sorted(
        comparison.loc[comparison["performance_crossover_task_within_tested_cpus"], "dataset"].unique()
    )
    workflow_crossovers = sorted(
        comparison.loc[comparison["performance_crossover_workflow_within_tested_cpus"], "dataset"].unique()
    )
    task_dataset_crossovers = sorted(
        comparison.loc[comparison["performance_crossover_task_across_tested_datasets"], "cpus"].unique()
    )
    workflow_dataset_crossovers = sorted(
        comparison.loc[comparison["performance_crossover_workflow_across_tested_datasets"], "cpus"].unique()
    )
    if task_crossovers:
        task_crossover_text = f"Task-runtime crossover observed for: {', '.join(task_crossovers)}."
    else:
        task_crossover_text = "No task-runtime performance crossover was observed."
    if workflow_crossovers:
        workflow_crossover_text = f"Workflow-runtime crossover observed for: {', '.join(workflow_crossovers)}."
    else:
        workflow_crossover_text = "No workflow-runtime performance crossover was observed."
    if task_dataset_crossovers:
        task_dataset_crossover_text = (
            "Task-runtime winner changed across datasets at CPU settings: "
            f"{', '.join(str(value) for value in task_dataset_crossovers)}."
        )
    else:
        task_dataset_crossover_text = "The task-runtime winner did not change across tested datasets."
    if workflow_dataset_crossovers:
        workflow_dataset_crossover_text = (
            "Workflow-runtime winner changed across datasets at CPU settings: "
            f"{', '.join(str(value) for value in workflow_dataset_crossovers)}."
        )
    else:
        workflow_dataset_crossover_text = "The workflow-runtime winner did not change across tested datasets."
    lines.extend(
        [
            "",
            f"Tested CPU settings: {cpu_text}. {task_crossover_text} {workflow_crossover_text}",
            f"{task_dataset_crossover_text} {workflow_dataset_crossover_text}",
            "These crossover statements apply only to the tested CPU settings and do not imply behaviour "
            "between or beyond them.",
            "Dataset differences combine matrix size, biology, and design, so a dataset crossover cannot be "
            "attributed to size alone.",
            "",
            "Task duration and end-to-end workflow duration are reported separately; neither implementation "
            "was required to win.",
        ]
    )
    (args.outdir / "benchmark_conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
