#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare R DESeq2 and PyDESeq2 result tables.")
    parser.add_argument("--dataset-name", required=True, help="Dataset label used in plots and provenance.")
    parser.add_argument("--output-prefix", required=True, help="Prefix used for comparison output files.")
    parser.add_argument("--r-results", required=True, help="R DESeq2 results TSV with gene_id column.")
    parser.add_argument("--pydeseq2-results", required=True, help="PyDESeq2 results TSV with gene_id column.")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Adjusted p-value significance threshold.")
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Requested rank cutoff; all genes tied at the (padj, pvalue) boundary are included.",
    )
    parser.add_argument("--r-runtime", help="Optional repeated R core-runtime TSV from dataset preparation.")
    parser.add_argument(
        "--pydeseq2-session-info",
        help="Optional legacy Python_sessionInfo.log core-runtime fallback.",
    )
    parser.add_argument(
        "--nextflow-trace",
        "--pydeseq2-trace",
        dest="nextflow_trace",
        help="Optional Nextflow trace file from the paired nf-core module run.",
    )
    parser.add_argument("--pydeseq2-runtime-seconds", type=float, help="Optional PyDESeq2 core seconds override.")
    parser.add_argument(
        "--pydeseq2-runtime",
        help="Optional PyDESeq2 runtime-profile TSV written by the current module.",
    )
    parser.add_argument("--r-task-seconds", type=float, help="Optional nf-core R module task seconds override.")
    parser.add_argument(
        "--pydeseq2-task-seconds",
        type=float,
        help="Optional nf-core PyDESeq2 module task seconds override.",
    )
    parser.add_argument("--benchmark-timings", help="Optional per-run benchmark timing TSV.")
    parser.add_argument(
        "--cpus",
        type=int,
        help=(
            "CPU setting to select from --benchmark-timings; required with --r-runtime and when the dataset "
            "has multiple CPU values."
        ),
    )
    parser.add_argument("--r-size-factors", help="Optional R DESeq2 size-factor TSV.")
    parser.add_argument("--pydeseq2-size-factors", help="Optional PyDESeq2 size-factor TSV.")
    parser.add_argument(
        "--r-normalised-counts",
        "--r-normalized-counts",
        dest="r_normalised_counts",
        help="Optional R DESeq2 normalised-count matrix TSV.",
    )
    parser.add_argument(
        "--pydeseq2-normalised-counts",
        "--pydeseq2-normalized-counts",
        dest="pydeseq2_normalised_counts",
        help="Optional PyDESeq2 normalised-count matrix TSV.",
    )
    parser.add_argument(
        "--r-vst-counts",
        "--r-vst",
        dest="r_vst_counts",
        help="Optional R DESeq2 VST matrix TSV.",
    )
    parser.add_argument(
        "--pydeseq2-vst-counts",
        "--pydeseq2-vst",
        dest="pydeseq2_vst_counts",
        help="Optional PyDESeq2 VST matrix TSV.",
    )
    args = parser.parse_args(argv)
    if args.top_n < 1:
        parser.error("--top-n must be at least 1")
    if args.cpus is not None and args.cpus < 1:
        parser.error("--cpus must be at least 1")
    if args.r_runtime and not args.benchmark_timings:
        parser.error("--benchmark-timings is required with --r-runtime")
    if args.r_runtime and args.cpus is None:
        parser.error("--cpus is required with --r-runtime")
    try:
        validate_paired_paths(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def validate_paired_paths(args):
    pairs = (
        ("size factors", args.r_size_factors, args.pydeseq2_size_factors),
        ("normalised counts", args.r_normalised_counts, args.pydeseq2_normalised_counts),
        ("VST counts", args.r_vst_counts, args.pydeseq2_vst_counts),
    )
    for label, r_path, py_path in pairs:
        if bool(r_path) != bool(py_path):
            raise ValueError(f"Both R and PyDESeq2 {label} inputs are required when either is supplied")


def read_results(path):
    frame = pd.read_csv(path, sep="\t", na_values=["NA", "NaN", ""], keep_default_na=True)
    if "gene_id" not in frame.columns:
        raise ValueError(f"{path} does not contain a gene_id column")
    if frame["gene_id"].isna().any():
        raise ValueError(f"{path} contains missing gene identifiers")
    if frame["gene_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate gene identifiers")
    return frame


def finite_pair(frame, left, right):
    values = frame[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
    return values


def correlation(frame, left, right, method):
    values = finite_pair(frame, left, right)
    if len(values) < 2:
        return np.nan
    return array_correlation(values[left].to_numpy(), values[right].to_numpy(), method)


def array_correlation(left, right, method="pearson"):
    left = np.asarray(left, dtype=float).ravel()
    right = np.asarray(right, dtype=float).ravel()
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if left.size < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return np.nan
    if method == "spearman":
        left = pd.Series(left).rank(method="average").to_numpy()
        right = pd.Series(right).rank(method="average").to_numpy()
    elif method != "pearson":
        raise ValueError(f"Unsupported correlation method: {method}")
    return float(np.corrcoef(left, right)[0, 1])


def significant_genes(frame, column, alpha):
    if column not in frame:
        return set()
    values = frame[["gene_id", column]].dropna()
    return set(values.loc[values[column] < alpha, "gene_id"])


def inclusive_top_ranked_genes(frame, n, padj_column="padj", pvalue_column="pvalue"):
    """Return the rank-N set without using gene identifiers to break boundary ties."""
    required = {"gene_id", padj_column, pvalue_column}
    if not required.issubset(frame.columns):
        missing = ", ".join(sorted(required.difference(frame.columns)))
        raise ValueError(f"Tie-aware ranking is missing required columns: {missing}")

    values = frame[["gene_id", padj_column, pvalue_column]].copy()
    values[padj_column] = pd.to_numeric(values[padj_column], errors="coerce")
    values[pvalue_column] = pd.to_numeric(values[pvalue_column], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna(subset=[padj_column, pvalue_column])
    values = values.sort_values([padj_column, pvalue_column], kind="mergesort").reset_index(drop=True)
    if values.empty:
        return {
            "genes": set(),
            "table": pd.DataFrame(
                columns=["gene_id", "padj", "pvalue", "rank_min", "rank_max", "boundary_member"]
            ),
            "eligible": 0,
            "included": 0,
            "boundary_padj": np.nan,
            "boundary_pvalue": np.nan,
            "boundary_tie_size": 0,
            "boundary_tie_expansion": 0,
        }

    boundary_index = min(n, len(values)) - 1
    boundary_padj = values.at[boundary_index, padj_column]
    boundary_pvalue = values.at[boundary_index, pvalue_column]
    before_boundary = (values[padj_column] < boundary_padj) | (
        (values[padj_column] == boundary_padj) & (values[pvalue_column] < boundary_pvalue)
    )
    at_boundary = (values[padj_column] == boundary_padj) & (values[pvalue_column] == boundary_pvalue)
    boundary_start = int(before_boundary.sum()) + 1
    boundary_end = boundary_start + int(at_boundary.sum()) - 1

    key_counts = values.groupby([padj_column, pvalue_column], sort=False)["gene_id"].transform("size")
    within_tie_position = values.groupby([padj_column, pvalue_column], sort=False).cumcount()
    values["rank_min"] = np.arange(1, len(values) + 1) - within_tie_position
    values["rank_max"] = values["rank_min"] + key_counts - 1
    included = values.loc[before_boundary | at_boundary].copy()
    included["boundary_member"] = at_boundary.loc[included.index].to_numpy()
    included = included.rename(columns={padj_column: "padj", pvalue_column: "pvalue"})

    return {
        "genes": set(included["gene_id"]),
        "table": included[["gene_id", "padj", "pvalue", "rank_min", "rank_max", "boundary_member"]],
        "eligible": int(len(values)),
        "included": int(len(included)),
        "boundary_padj": float(boundary_padj),
        "boundary_pvalue": float(boundary_pvalue),
        "boundary_tie_size": int(at_boundary.sum()),
        "boundary_tie_expansion": int(len(included) - min(n, len(values))),
        "boundary_rank_min": boundary_start,
        "boundary_rank_max": boundary_end,
    }


def top_genes(frame, column, n):
    """Compatibility wrapper returning the inclusive (padj, pvalue)-ranked set."""
    if column != "padj":
        raise ValueError("Tie-aware ranking requires padj as the primary rank column")
    return inclusive_top_ranked_genes(frame, n)["genes"]


def set_jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def abs_difference_metrics(left, right):
    left = np.asarray(left, dtype=float).ravel()
    right = np.asarray(right, dtype=float).ravel()
    finite = np.isfinite(left) & np.isfinite(right)
    differences = np.abs(left[finite] - right[finite])
    if differences.size == 0:
        return {
            "finite_pairs": 0,
            "abs_diff_median": np.nan,
            "abs_diff_p95": np.nan,
            "abs_diff_p99": np.nan,
            "abs_diff_max": np.nan,
        }
    return {
        "finite_pairs": int(differences.size),
        "abs_diff_median": float(np.quantile(differences, 0.5)),
        "abs_diff_p95": float(np.quantile(differences, 0.95)),
        "abs_diff_p99": float(np.quantile(differences, 0.99)),
        "abs_diff_max": float(differences.max()),
    }


def shared_numeric_result_columns(r, py):
    shared = sorted((set(r.columns) & set(py.columns)) - {"gene_id"})
    numeric = []
    for column in shared:
        r_values = pd.to_numeric(r[column], errors="coerce")
        py_values = pd.to_numeric(py[column], errors="coerce")
        r_nonmissing = r[column].notna().sum()
        py_nonmissing = py[column].notna().sum()
        if r_values.notna().sum() == r_nonmissing and py_values.notna().sum() == py_nonmissing:
            numeric.append(column)
    return numeric


def numeric_difference_diagnostics(r, py, joined, largest_n=20):
    summary_rows = []
    na_rows = []
    largest_rows = []
    for column in shared_numeric_result_columns(r, py):
        left_column = f"r_{column}"
        right_column = f"py_{column}"
        values = joined[["gene_id", left_column, right_column]].copy()
        values[left_column] = pd.to_numeric(values[left_column], errors="coerce")
        values[right_column] = pd.to_numeric(values[right_column], errors="coerce")
        r_na = values[left_column].isna()
        py_na = values[right_column].isna()
        mismatched_na = r_na != py_na
        for row in values.loc[mismatched_na].itertuples(index=False, name=None):
            na_rows.append(
                {
                    "gene_id": row[0],
                    "column": column,
                    "r_value": row[1],
                    "pydeseq2_value": row[2],
                    "r_is_na": bool(pd.isna(row[1])),
                    "pydeseq2_is_na": bool(pd.isna(row[2])),
                }
            )

        metrics = abs_difference_metrics(values[left_column], values[right_column])
        summary_rows.append(
            {
                "column": column,
                "shared_rows": int(len(values)),
                "r_na": int(r_na.sum()),
                "pydeseq2_na": int(py_na.sum()),
                "na_mask_disagreements": int(mismatched_na.sum()),
                "pearson": array_correlation(values[left_column], values[right_column], "pearson"),
                "spearman": array_correlation(values[left_column], values[right_column], "spearman"),
                **metrics,
            }
        )

        finite = np.isfinite(values[left_column]) & np.isfinite(values[right_column])
        differences = values.loc[finite].copy()
        differences["abs_difference"] = np.abs(differences[left_column] - differences[right_column])
        differences = differences.sort_values("abs_difference", ascending=False, kind="mergesort").head(largest_n)
        output_values = differences[["gene_id", left_column, right_column, "abs_difference"]]
        for diagnostic_rank, row in enumerate(output_values.itertuples(index=False, name=None), start=1):
            largest_rows.append(
                {
                    "column": column,
                    "diagnostic_rank": diagnostic_rank,
                    "gene_id": row[0],
                    "r_value": row[1],
                    "pydeseq2_value": row[2],
                    "abs_difference": row[3],
                }
            )

    summary_columns = [
        "column",
        "shared_rows",
        "finite_pairs",
        "r_na",
        "pydeseq2_na",
        "na_mask_disagreements",
        "pearson",
        "spearman",
        "abs_diff_median",
        "abs_diff_p95",
        "abs_diff_p99",
        "abs_diff_max",
    ]
    na_columns = ["gene_id", "column", "r_value", "pydeseq2_value", "r_is_na", "pydeseq2_is_na"]
    largest_columns = ["column", "diagnostic_rank", "gene_id", "r_value", "pydeseq2_value", "abs_difference"]
    return (
        pd.DataFrame(summary_rows, columns=summary_columns),
        pd.DataFrame(na_rows, columns=na_columns),
        pd.DataFrame(largest_rows, columns=largest_columns),
    )


def sign_disagreement_table(joined):
    columns = [
        "gene_id",
        "r_log2FoldChange",
        "pydeseq2_log2FoldChange",
        "r_sign",
        "pydeseq2_sign",
    ]
    if not {"r_log2FoldChange", "py_log2FoldChange"}.issubset(joined.columns):
        return pd.DataFrame(columns=columns)
    values = joined[["gene_id", "r_log2FoldChange", "py_log2FoldChange"]].copy()
    finite = np.isfinite(values["r_log2FoldChange"]) & np.isfinite(values["py_log2FoldChange"])
    values = values.loc[finite].copy()
    values["r_sign"] = np.sign(values["r_log2FoldChange"]).astype(int)
    values["pydeseq2_sign"] = np.sign(values["py_log2FoldChange"]).astype(int)
    values = values.loc[values["r_sign"] != values["pydeseq2_sign"]]
    return values.rename(columns={"py_log2FoldChange": "pydeseq2_log2FoldChange"})[columns]


def significance_disagreement_table(r, py, alpha):
    columns = ["gene_id", "r_padj", "pydeseq2_padj", "r_significant", "pydeseq2_significant"]
    r_values = r[["gene_id", "padj"]].rename(columns={"padj": "r_padj"})
    py_values = py[["gene_id", "padj"]].rename(columns={"padj": "pydeseq2_padj"})
    values = r_values.merge(py_values, on="gene_id", how="outer", validate="one_to_one")
    values["r_significant"] = values["r_padj"].notna() & (values["r_padj"] < alpha)
    values["pydeseq2_significant"] = values["pydeseq2_padj"].notna() & (values["pydeseq2_padj"] < alpha)
    return values.loc[values["r_significant"] != values["pydeseq2_significant"], columns]


def parse_duration_seconds(value):
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        pass

    total = 0.0
    matched = False
    for amount, unit in re.findall(r"([0-9]*\.?[0-9]+)\s*(ms|[dhms])", text):
        matched = True
        scale = {"d": 86400, "h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
        total += float(amount) * scale
    return total if matched else np.nan


def read_r_runtime(path, cpus, timings=None):
    if not path:
        return None
    if cpus is None:
        raise ValueError("--cpus is required with --r-runtime")
    if timings is None or timings.empty:
        raise ValueError("--r-runtime requires a selected benchmark timing table")
    runtime = pd.read_csv(path, sep="\t")
    timing_metadata = {
        "profile",
        "r_version",
        "deseq2_version",
        "host_system",
        "host_machine",
        "genes",
        "samples",
        "design",
        "contrast",
    }
    missing_timing_metadata = timing_metadata.difference(timings.columns)
    if missing_timing_metadata:
        raise ValueError(
            "R core timing cannot be validated because the benchmark table is missing: "
            f"{', '.join(sorted(missing_timing_metadata))}"
        )
    profiles = timings["profile"].dropna().astype(str).str.strip().unique()
    if len(profiles) != 1 or profiles[0] != "local" or timings["profile"].isna().any():
        raise ValueError("--r-runtime can be combined only with a matched local benchmark table")

    runtime_metadata = {
        "timer_runtime",
        "timer_runtime_version",
        "timer_engine",
        "timer_engine_version",
        "timer_host_system",
        "timer_host_machine",
        "timer_dataset",
        "timer_genes",
        "timer_samples",
        "timer_design",
        "timer_contrast",
    }
    missing_runtime_metadata = runtime_metadata.difference(runtime.columns)
    if missing_runtime_metadata:
        raise ValueError(
            "R runtime TSV is missing timer provenance: "
            f"{', '.join(sorted(missing_runtime_metadata))}"
        )
    metadata = {}
    for column in runtime_metadata:
        if runtime[column].isna().any():
            raise ValueError(f"R runtime {column} must be non-empty and constant")
        values = runtime[column].astype(str).str.strip().unique()
        if len(values) != 1 or not values[0]:
            raise ValueError(f"R runtime {column} must be non-empty and constant")
        metadata[column] = values[0]

    r_timings = timings.loc[timings["implementation"].eq("r")]
    if r_timings.empty:
        raise ValueError("Selected benchmark timing table has no R rows for runtime validation")
    expected = {
        "timer_runtime": "R",
        "timer_runtime_version": r_timings["r_version"],
        "timer_engine": "DESeq2",
        "timer_engine_version": r_timings["deseq2_version"],
        "timer_host_system": timings["host_system"],
        "timer_host_machine": timings["host_machine"],
        "timer_dataset": timings["dataset"],
        "timer_genes": timings["genes"],
        "timer_samples": timings["samples"],
        "timer_design": timings["design"],
        "timer_contrast": timings["contrast"],
    }
    for column, expected_values in expected.items():
        if isinstance(expected_values, str):
            expected_value = expected_values
        else:
            values = expected_values.dropna().astype(str).str.strip().unique()
            if len(values) != 1 or expected_values.isna().any():
                raise ValueError(f"Benchmark {column} provenance must be non-empty and constant")
            expected_value = values[0]
        observed_value = metadata[column]
        if column == "timer_dataset":
            matches = observed_value.casefold() == expected_value.casefold()
        elif column in {"timer_genes", "timer_samples"}:
            try:
                observed_number = float(observed_value)
                expected_number = float(expected_value)
            except ValueError as error:
                raise ValueError(f"R runtime {column} and benchmark identity must be numeric") from error
            matches = (
                observed_number.is_integer()
                and expected_number.is_integer()
                and observed_number > 0
                and observed_number == expected_number
            )
        else:
            matches = observed_value == expected_value
        if not matches:
            raise ValueError(f"R runtime {column} does not match the benchmark provenance")
    if not {"step", "elapsed_seconds"}.issubset(runtime.columns):
        raise ValueError(f"{path} must contain step and elapsed_seconds columns")
    if "cpus" not in runtime.columns:
        raise ValueError(f"{path} must contain a cpus column when --cpus is supplied")
    runtime_cpus = pd.to_numeric(runtime["cpus"], errors="coerce")
    if runtime_cpus.isna().any() or not runtime_cpus.mod(1).eq(0).all() or (runtime_cpus < 1).any():
        raise ValueError(f"{path} contains invalid CPU values")
    runtime = runtime.loc[runtime_cpus.astype(int) == cpus]
    if runtime.empty:
        raise ValueError(f"{path} has no {cpus}-CPU timing rows")
    core_values = pd.to_numeric(
        runtime.loc[runtime["step"] == "deseq_and_results", "elapsed_seconds"],
        errors="coerce",
    )
    if core_values.empty or core_values.isna().any() or not np.isfinite(core_values).all():
        raise ValueError(f"{path} has no finite deseq_and_results timings for the selected CPU setting")
    if (core_values < 0).any():
        raise ValueError(f"{path} contains negative deseq_and_results timing")
    return float(core_values.median())


def read_benchmark_timings(path, dataset_name, cpus):
    if not path:
        return pd.DataFrame()
    timings = pd.read_csv(path, sep="\t")
    required = {"dataset", "implementation", "cpus", "repetition", "wall_seconds"}
    missing = required.difference(timings.columns)
    if missing:
        raise ValueError(f"Benchmark timing table is missing columns: {', '.join(sorted(missing))}")
    selected = timings.loc[timings["dataset"].astype(str).str.lower() == dataset_name.lower()].copy()
    if selected.empty:
        raise ValueError(f"Benchmark timing table has no rows for dataset '{dataset_name}'")
    available_cpus = sorted(pd.to_numeric(selected["cpus"], errors="coerce").dropna().unique())
    if cpus is None and len(available_cpus) > 1:
        cpu_list = ", ".join(str(int(value)) if float(value).is_integer() else str(value) for value in available_cpus)
        raise ValueError(
            f"--cpus is required because dataset '{dataset_name}' has multiple CPU values: {cpu_list}"
        )
    if cpus is not None:
        selected = selected.loc[pd.to_numeric(selected["cpus"], errors="coerce") == cpus]
        if selected.empty:
            raise ValueError(f"Benchmark timing table has no rows for dataset '{dataset_name}' at {cpus} CPUs")
    return selected


def timing_median(timings, implementation, column):
    if timings.empty or column not in timings.columns:
        return None
    values = pd.to_numeric(
        timings.loc[timings["implementation"] == implementation, column],
        errors="coerce",
    ).dropna()
    return float(values.median()) if not values.empty else None


def timing_value(timings, implementation, column, reducer):
    if timings.empty or column not in timings.columns:
        return None
    values = pd.to_numeric(
        timings.loc[timings["implementation"] == implementation, column],
        errors="coerce",
    ).dropna()
    if values.empty:
        return None
    return float(reducer(values))


def results_are_deterministic(timings, implementation):
    if timings.empty or "result_sha256" not in timings.columns:
        return None
    values = timings.loc[timings["implementation"] == implementation, "result_sha256"].dropna()
    return bool(not values.empty and values.nunique() == 1)


def read_pydeseq2_core_runtime(session_info_path, override):
    if override is not None:
        return float(override), None, None
    if not session_info_path:
        return None, None, None
    text = Path(session_info_path).read_text(encoding="utf-8")
    payload = json.loads(text[text.index("{") :])
    options = payload.get("options", {})
    runtime = options.get("runtime_seconds", {})
    value = runtime.get("pydeseq2_core_deseq2_and_stats")
    return (
        float(value) if value is not None else None,
        options.get("cores"),
        options.get("vs_method"),
    )


def read_pydeseq2_runtime(path):
    runtime = pd.read_csv(path, sep="\t")
    required = {"step", "elapsed_seconds"}
    if not required.issubset(runtime.columns):
        raise ValueError(f"{path} must contain step and elapsed_seconds columns")
    if runtime["step"].duplicated().any():
        raise ValueError(f"{path} contains duplicate runtime steps")
    elapsed = pd.to_numeric(runtime["elapsed_seconds"], errors="raise")
    phases = dict(zip(runtime["step"], elapsed, strict=True))
    if "pydeseq2_core_deseq2_and_stats" in phases:
        core_seconds = float(phases["pydeseq2_core_deseq2_and_stats"])
        source = "pydeseq2_runtime_tsv:pydeseq2_core_deseq2_and_stats"
    elif {"core_dds_deseq2", "stats_summary"}.issubset(phases):
        core_seconds = float(phases["core_dds_deseq2"] + phases["stats_summary"])
        source = "pydeseq2_runtime_tsv:core_dds_deseq2+stats_summary"
    else:
        raise ValueError(f"{path} does not contain a supported PyDESeq2 core timing")
    if not np.isfinite(core_seconds) or core_seconds < 0:
        raise ValueError(f"{path} contains an invalid PyDESeq2 core timing")
    return core_seconds, source


def read_trace_runtime(trace_path, process_name):
    if not trace_path:
        return None
    trace = pd.read_csv(trace_path, sep="\t")
    process_column = "process" if "process" in trace.columns else "name" if "name" in trace.columns else None
    if process_column is None:
        return None
    process = trace[process_column].astype(str)
    rows = trace.loc[process.str.match(rf"(^|.*:){re.escape(process_name)}(\s|$)", na=False)]
    if len(rows) == 0:
        return None
    for column in ("realtime", "duration"):
        if column in rows.columns:
            values = rows[column].map(parse_duration_seconds).dropna()
            if len(values) > 0:
                return float(values.sum())
    return None


def runtime_ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def runtime_value_available(value):
    return value is not None and np.isfinite(value)


def resolve_runtime_inputs(args, timings):
    numeric_overrides = (
        ("--pydeseq2-runtime-seconds", args.pydeseq2_runtime_seconds, True),
        ("--r-task-seconds", args.r_task_seconds, False),
        ("--pydeseq2-task-seconds", args.pydeseq2_task_seconds, False),
    )
    for option, value, allow_zero in numeric_overrides:
        if value is None:
            continue
        valid = np.isfinite(value) and (value >= 0 if allow_zero else value > 0)
        if not valid:
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{option} must be a finite {qualifier} value")

    r_benchmark_core = timing_median(timings, "r", "core_seconds")
    r_file_core = read_r_runtime(args.r_runtime, args.cpus, timings) if args.r_runtime else None
    if runtime_value_available(r_file_core):
        r_core_runtime_seconds = r_file_core
        r_core_runtime_source = "r_runtime_file"
    elif runtime_value_available(r_benchmark_core):
        r_core_runtime_seconds = r_benchmark_core
        r_core_runtime_source = "benchmark_timings"
    else:
        r_core_runtime_seconds = None
        r_core_runtime_source = None

    pydeseq2_benchmark_core = timing_median(timings, "pydeseq2", "core_seconds")
    if runtime_value_available(args.pydeseq2_runtime_seconds):
        pydeseq2_core_runtime_seconds = float(args.pydeseq2_runtime_seconds)
        pydeseq2_core_runtime_source = "cli_override"
        pydeseq2_cores = None
        pydeseq2_vs_method = None
    elif runtime_value_available(pydeseq2_benchmark_core):
        pydeseq2_core_runtime_seconds = pydeseq2_benchmark_core
        pydeseq2_core_runtime_source = "benchmark_timings"
        pydeseq2_cores = None
        pydeseq2_vs_method = None
    elif args.pydeseq2_runtime:
        pydeseq2_core_runtime_seconds, pydeseq2_core_runtime_source = read_pydeseq2_runtime(
            args.pydeseq2_runtime
        )
        pydeseq2_cores = None
        pydeseq2_vs_method = None
    else:
        session_core, pydeseq2_cores, pydeseq2_vs_method = read_pydeseq2_core_runtime(
            args.pydeseq2_session_info,
            None,
        )
        if session_core is not None and (not np.isfinite(session_core) or session_core < 0):
            raise ValueError("PyDESeq2 session core runtime must be finite and non-negative")
        if runtime_value_available(session_core):
            if pydeseq2_cores is not None:
                try:
                    session_cpus = float(pydeseq2_cores)
                except (TypeError, ValueError) as error:
                    raise ValueError("PyDESeq2 session CPU count must be a positive integer") from error
                if not np.isfinite(session_cpus) or not session_cpus.is_integer() or session_cpus < 1:
                    raise ValueError("PyDESeq2 session CPU count must be a positive integer")
                if args.cpus is not None and int(session_cpus) != args.cpus:
                    raise ValueError(
                        "PyDESeq2 session CPU count does not match the selected benchmark CPU setting"
                    )
            pydeseq2_core_runtime_seconds = session_core
            pydeseq2_core_runtime_source = "pydeseq2_session_info"
        else:
            pydeseq2_core_runtime_seconds = None
            pydeseq2_core_runtime_source = None
    if args.cpus is not None:
        pydeseq2_cores = args.cpus

    r_benchmark_task = timing_median(timings, "r", "wall_seconds")
    if runtime_value_available(args.r_task_seconds):
        r_nfcore_task_seconds = float(args.r_task_seconds)
        r_nfcore_task_source = "cli_override"
    elif runtime_value_available(r_benchmark_task):
        r_nfcore_task_seconds = r_benchmark_task
        r_nfcore_task_source = "benchmark_timings"
    else:
        r_nfcore_task_seconds = read_trace_runtime(args.nextflow_trace, "DESEQ2_DIFFERENTIAL")
        r_nfcore_task_source = "nextflow_trace" if runtime_value_available(r_nfcore_task_seconds) else None

    pydeseq2_benchmark_task = timing_median(timings, "pydeseq2", "wall_seconds")
    if runtime_value_available(args.pydeseq2_task_seconds):
        pydeseq2_nfcore_task_seconds = float(args.pydeseq2_task_seconds)
        pydeseq2_nfcore_task_source = "cli_override"
    elif runtime_value_available(pydeseq2_benchmark_task):
        pydeseq2_nfcore_task_seconds = pydeseq2_benchmark_task
        pydeseq2_nfcore_task_source = "benchmark_timings"
    else:
        pydeseq2_nfcore_task_seconds = read_trace_runtime(args.nextflow_trace, "PYDESEQ2_DIFFERENTIAL")
        pydeseq2_nfcore_task_source = (
            "nextflow_trace" if runtime_value_available(pydeseq2_nfcore_task_seconds) else None
        )

    return {
        "r_core_runtime_seconds": r_core_runtime_seconds,
        "r_core_runtime_source": r_core_runtime_source,
        "pydeseq2_core_runtime_seconds": pydeseq2_core_runtime_seconds,
        "pydeseq2_core_runtime_source": pydeseq2_core_runtime_source,
        "pydeseq2_cores": pydeseq2_cores,
        "pydeseq2_vs_method": pydeseq2_vs_method,
        "r_nfcore_task_seconds": r_nfcore_task_seconds,
        "r_nfcore_task_source": r_nfcore_task_source,
        "pydeseq2_nfcore_task_seconds": pydeseq2_nfcore_task_seconds,
        "pydeseq2_nfcore_task_source": pydeseq2_nfcore_task_source,
    }


def json_value(value):
    if value is None:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def read_size_factors(path):
    frame = pd.read_csv(path, sep="\t", na_values=["NA", "NaN", ""], keep_default_na=True)
    required = {"sample", "sizeFactor"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain sample and sizeFactor columns")
    if frame["sample"].isna().any():
        raise ValueError(f"{path} contains missing sample identifiers")
    if frame["sample"].duplicated().any():
        raise ValueError(f"{path} contains duplicate sample identifiers")
    values = pd.to_numeric(frame["sizeFactor"], errors="raise")
    return pd.Series(values.to_numpy(), index=frame["sample"].astype(str), name="sizeFactor")


def compare_size_factors(r_values, py_values):
    r_samples = set(r_values.index)
    py_samples = set(py_values.index)
    shared_samples = sorted(r_samples & py_samples)
    r_aligned = r_values.loc[shared_samples].to_numpy(dtype=float)
    py_aligned = py_values.loc[shared_samples].to_numpy(dtype=float)
    metrics = {
        "size_factors_r_samples": len(r_samples),
        "size_factors_pydeseq2_samples": len(py_samples),
        "size_factors_shared_samples": len(shared_samples),
        "size_factors_r_only_samples": len(r_samples - py_samples),
        "size_factors_pydeseq2_only_samples": len(py_samples - r_samples),
        "size_factors_pearson": array_correlation(r_aligned, py_aligned, "pearson"),
        "size_factors_spearman": array_correlation(r_aligned, py_aligned, "spearman"),
        **{f"size_factors_{key}": value for key, value in abs_difference_metrics(r_aligned, py_aligned).items()},
    }
    comparison = pd.DataFrame(
        {
            "sample": shared_samples,
            "r_sizeFactor": r_aligned,
            "pydeseq2_sizeFactor": py_aligned,
            "difference": py_aligned - r_aligned,
            "abs_difference": np.abs(py_aligned - r_aligned),
        }
    )
    return metrics, comparison


def read_labelled_matrix(path):
    frame = pd.read_csv(path, sep="\t", na_values=["NA", "NaN", ""], keep_default_na=True)
    if "gene_id" not in frame.columns:
        raise ValueError(f"{path} does not contain a gene_id column")
    if frame["gene_id"].isna().any():
        raise ValueError(f"{path} contains missing gene identifiers")
    if frame["gene_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate gene identifiers")
    sample_columns = [column for column in frame.columns if column != "gene_id"]
    if not sample_columns:
        raise ValueError(f"{path} does not contain sample columns")
    matrix = frame.set_index(frame["gene_id"].astype(str))[sample_columns]
    matrix.index.name = "gene_id"
    return matrix.apply(pd.to_numeric, errors="raise")


def align_labelled_matrices(r_matrix, py_matrix):
    r_genes = set(r_matrix.index)
    py_genes = set(py_matrix.index)
    r_samples = set(r_matrix.columns)
    py_samples = set(py_matrix.columns)
    shared_genes = sorted(r_genes & py_genes)
    shared_samples = sorted(r_samples & py_samples)
    counts = {
        "r_genes": len(r_genes),
        "pydeseq2_genes": len(py_genes),
        "shared_genes": len(shared_genes),
        "r_only_genes": len(r_genes - py_genes),
        "pydeseq2_only_genes": len(py_genes - r_genes),
        "r_samples": len(r_samples),
        "pydeseq2_samples": len(py_samples),
        "shared_samples": len(shared_samples),
        "r_only_samples": len(r_samples - py_samples),
        "pydeseq2_only_samples": len(py_samples - r_samples),
    }
    return (
        r_matrix.loc[shared_genes, shared_samples],
        py_matrix.loc[shared_genes, shared_samples],
        counts,
    )


def compare_aligned_matrices(label, r_matrix, py_matrix, alignment_counts):
    r_values = r_matrix.to_numpy(dtype=float)
    py_values = py_matrix.to_numpy(dtype=float)
    metrics = {f"{label}_{key}": value for key, value in alignment_counts.items()}
    metrics.update(
        {
            f"{label}_flattened_pearson": array_correlation(r_values, py_values, "pearson"),
            f"{label}_flattened_spearman": array_correlation(r_values, py_values, "spearman"),
            **{
                f"{label}_{key}": value
                for key, value in abs_difference_metrics(r_values, py_values).items()
            },
        }
    )

    rows = []
    for sample in r_matrix.columns:
        r_sample = r_matrix[sample].to_numpy(dtype=float)
        py_sample = py_matrix[sample].to_numpy(dtype=float)
        rows.append(
            {
                "sample": sample,
                "pearson": array_correlation(r_sample, py_sample, "pearson"),
                "spearman": array_correlation(r_sample, py_sample, "spearman"),
                **abs_difference_metrics(r_sample, py_sample),
            }
        )
    per_sample_columns = [
        "sample",
        "finite_pairs",
        "pearson",
        "spearman",
        "abs_diff_median",
        "abs_diff_p95",
        "abs_diff_p99",
        "abs_diff_max",
    ]
    return metrics, pd.DataFrame(rows, columns=per_sample_columns)


def pca_scores(matrix):
    values = np.asarray(matrix, dtype=float).T
    centered = values - values.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :2] * singular_values[:2]
    if scores.shape[1] < 2:
        scores = np.pad(scores, ((0, 0), (0, 2 - scores.shape[1])))
    for component in range(2):
        if np.any(scores[:, component]):
            anchor = np.argmax(np.abs(scores[:, component]))
            if scores[anchor, component] < 0:
                scores[:, component] *= -1
    sum_squares = float(np.sum(singular_values**2))
    explained = np.zeros(2, dtype=float)
    if sum_squares > 0:
        available = min(2, len(singular_values))
        explained[:available] = singular_values[:available] ** 2 / sum_squares
    else:
        explained[:] = np.nan
    return scores, explained


def pairwise_distances(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.array([], dtype=float)
    squared_norms = np.sum(values**2, axis=1)
    squared_distances = squared_norms[:, None] + squared_norms[None, :] - 2 * (values @ values.T)
    row_indices, column_indices = np.triu_indices(len(values), k=1)
    return np.sqrt(np.maximum(squared_distances[row_indices, column_indices], 0))


def align_pca_scores(reference, target):
    reference_centered = reference - reference.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)
    reference_norm = np.linalg.norm(reference_centered)
    target_norm = np.linalg.norm(target_centered)
    if reference_norm == 0 or target_norm == 0:
        return target_centered, {
            "vst_pca_procrustes_disparity": np.nan,
            "vst_pca_procrustes_rmse": np.nan,
            "vst_pca_procrustes_pearson": np.nan,
        }
    reference_scaled = reference_centered / reference_norm
    target_scaled = target_centered / target_norm
    u, _, vt = np.linalg.svd(target_scaled.T @ reference_scaled, full_matrices=False)
    aligned_scaled = target_scaled @ (u @ vt)
    aligned = aligned_scaled * reference_norm + reference.mean(axis=0, keepdims=True)
    differences = aligned_scaled - reference_scaled
    return aligned, {
        "vst_pca_procrustes_disparity": float(np.sum(differences**2)),
        "vst_pca_procrustes_rmse": float(np.sqrt(np.mean(differences**2))),
        "vst_pca_procrustes_pearson": array_correlation(reference_scaled, aligned_scaled, "pearson"),
    }


def compare_vst_geometry(r_matrix, py_matrix):
    finite_genes = np.isfinite(r_matrix.to_numpy(dtype=float)).all(axis=1) & np.isfinite(
        py_matrix.to_numpy(dtype=float)
    ).all(axis=1)
    r_complete = r_matrix.loc[finite_genes]
    py_complete = py_matrix.loc[finite_genes]
    coordinate_columns = [
        "sample",
        "r_PC1",
        "r_PC2",
        "pydeseq2_PC1",
        "pydeseq2_PC2",
        "pydeseq2_aligned_PC1",
        "pydeseq2_aligned_PC2",
    ]
    if r_complete.empty or r_complete.shape[1] < 2:
        metrics = {
            "vst_pca_complete_genes": int(len(r_complete)),
            "vst_r_pc1_explained_variance": np.nan,
            "vst_r_pc2_explained_variance": np.nan,
            "vst_pydeseq2_pc1_explained_variance": np.nan,
            "vst_pydeseq2_pc2_explained_variance": np.nan,
            "vst_sample_distance_pearson": np.nan,
            "vst_sample_distance_spearman": np.nan,
            "vst_pca_distance_pearson": np.nan,
            "vst_pca_distance_spearman": np.nan,
            "vst_pca_procrustes_disparity": np.nan,
            "vst_pca_procrustes_rmse": np.nan,
            "vst_pca_procrustes_pearson": np.nan,
        }
        return metrics, pd.DataFrame(columns=coordinate_columns)

    r_scores, r_explained = pca_scores(r_complete.to_numpy(dtype=float))
    py_scores, py_explained = pca_scores(py_complete.to_numpy(dtype=float))
    py_aligned, procrustes_metrics = align_pca_scores(r_scores, py_scores)
    r_sample_distances = pairwise_distances(r_complete.to_numpy(dtype=float).T)
    py_sample_distances = pairwise_distances(py_complete.to_numpy(dtype=float).T)
    r_pca_distances = pairwise_distances(r_scores)
    py_pca_distances = pairwise_distances(py_scores)
    metrics = {
        "vst_pca_complete_genes": int(len(r_complete)),
        "vst_r_pc1_explained_variance": float(r_explained[0]),
        "vst_r_pc2_explained_variance": float(r_explained[1]),
        "vst_pydeseq2_pc1_explained_variance": float(py_explained[0]),
        "vst_pydeseq2_pc2_explained_variance": float(py_explained[1]),
        "vst_sample_distance_pearson": array_correlation(r_sample_distances, py_sample_distances, "pearson"),
        "vst_sample_distance_spearman": array_correlation(r_sample_distances, py_sample_distances, "spearman"),
        "vst_pca_distance_pearson": array_correlation(r_pca_distances, py_pca_distances, "pearson"),
        "vst_pca_distance_spearman": array_correlation(r_pca_distances, py_pca_distances, "spearman"),
        **procrustes_metrics,
    }
    coordinates = pd.DataFrame(
        {
            "sample": r_complete.columns,
            "r_PC1": r_scores[:, 0],
            "r_PC2": r_scores[:, 1],
            "pydeseq2_PC1": py_scores[:, 0],
            "pydeseq2_PC2": py_scores[:, 1],
            "pydeseq2_aligned_PC1": py_aligned[:, 0],
            "pydeseq2_aligned_PC2": py_aligned[:, 1],
        }
    )
    return metrics, coordinates[coordinate_columns]


def write_vst_pca_plot(coordinates, metrics, dataset_name, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    if coordinates.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "Insufficient finite data for PCA", ha="center", va="center")
            ax.set_axis_off()
    else:
        colors = np.arange(len(coordinates))
        panels = (
            (axes[0], "r_PC1", "r_PC2", "r", "R DESeq2"),
            (axes[1], "pydeseq2_PC1", "pydeseq2_PC2", "pydeseq2", "PyDESeq2"),
        )
        for ax, x_column, y_column, implementation, title in panels:
            ax.scatter(
                coordinates[x_column],
                coordinates[y_column],
                c=colors,
                cmap="viridis",
                s=30,
                linewidths=0,
            )
            if len(coordinates) <= 20:
                for row in coordinates.itertuples(index=False):
                    ax.annotate(
                        row.sample,
                        (getattr(row, x_column), getattr(row, y_column)),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=7,
                    )
            ax.set_xlabel(f"PC1 ({metrics[f'vst_{implementation}_pc1_explained_variance']:.1%})")
            ax.set_ylabel(f"PC2 ({metrics[f'vst_{implementation}_pc2_explained_variance']:.1%})")
            ax.set_title(title)
            ax.grid(True, color="0.9", linewidth=0.8)
    fig.suptitle(f"{dataset_name} VST PCA on aligned genes and samples")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def compare_transform_inputs(args, outdir):
    prefix = args.output_prefix
    summary = {}
    if args.r_size_factors:
        r_size_factors = read_size_factors(args.r_size_factors)
        py_size_factors = read_size_factors(args.pydeseq2_size_factors)
        metrics, comparison = compare_size_factors(r_size_factors, py_size_factors)
        summary.update(metrics)
        comparison.to_csv(outdir / f"{prefix}_size_factor_comparison.tsv", sep="\t", index=False, na_rep="NA")

    matrix_pairs = (
        (
            "normalised_counts",
            args.r_normalised_counts,
            args.pydeseq2_normalised_counts,
            f"{prefix}_normalised_counts_per_sample_correlations.tsv",
        ),
        (
            "vst",
            args.r_vst_counts,
            args.pydeseq2_vst_counts,
            f"{prefix}_vst_per_sample_correlations.tsv",
        ),
    )
    for label, r_path, py_path, output_name in matrix_pairs:
        if not r_path:
            continue
        r_matrix = read_labelled_matrix(r_path)
        py_matrix = read_labelled_matrix(py_path)
        r_aligned, py_aligned, alignment_counts = align_labelled_matrices(r_matrix, py_matrix)
        metrics, per_sample = compare_aligned_matrices(label, r_aligned, py_aligned, alignment_counts)
        summary.update(metrics)
        per_sample.to_csv(outdir / output_name, sep="\t", index=False, na_rep="NA")

        if label == "vst":
            r_centered = r_aligned.sub(r_aligned.mean(axis=1), axis=0)
            py_centered = py_aligned.sub(py_aligned.mean(axis=1), axis=0)
            summary["vst_gene_centered_pearson"] = array_correlation(
                r_centered.to_numpy(), py_centered.to_numpy(), "pearson"
            )
            summary["vst_gene_centered_spearman"] = array_correlation(
                r_centered.to_numpy(), py_centered.to_numpy(), "spearman"
            )
            geometry_metrics, coordinates = compare_vst_geometry(r_aligned, py_aligned)
            summary.update(geometry_metrics)
            coordinates.to_csv(
                outdir / f"{prefix}_vst_pca_coordinates.tsv",
                sep="\t",
                index=False,
                na_rep="NA",
            )
            write_vst_pca_plot(
                coordinates,
                geometry_metrics,
                args.dataset_name,
                outdir / f"{prefix}_vst_pca.png",
            )

    if summary:
        pd.DataFrame([summary]).to_csv(
            outdir / f"{prefix}_transform_summary.tsv",
            sep="\t",
            index=False,
            na_rep="NA",
        )
    return summary


def add_identity_line(ax, values):
    finite = values.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return
    low = min(finite.min())
    high = max(finite.max())
    ax.plot([low, high], [low, high], color="0.35", linewidth=1, linestyle="--")


def stats_label(summary, keys):
    lines = []
    for label, key, fmt in keys:
        value = summary.get(key)
        if value is None or pd.isna(value):
            text = "NA"
        elif fmt == "int":
            text = f"{int(value)}"
        elif fmt == "float":
            text = f"{float(value):.3f}"
        else:
            text = str(value)
        lines.append(f"{label}: {text}")
    return "\n".join(lines)


def add_stats_inset(ax, text):
    if not text:
        return
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )


def write_scatter(frame, left, right, xlabel, ylabel, title, path, transform=None, inset_text=None):
    values = finite_pair(frame, left, right)
    if transform is not None:
        values = values.assign(**{left: transform(values[left]), right: transform(values[right])})
        values = values.replace([np.inf, -np.inf], np.nan).dropna()

    fig, ax = plt.subplots(figsize=(5, 5))
    if len(values) > 0:
        ax.scatter(values[left], values[right], s=8, alpha=0.45, linewidths=0)
        add_identity_line(ax, values[[left, right]])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="0.9", linewidth=0.8)
    add_stats_inset(ax, inset_text)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_rank_plot(frame, dataset_name, outpath, inset_text):
    values = frame[["gene_id", "r_padj", "py_padj"]].replace([np.inf, -np.inf], np.nan).dropna()
    values = values.assign(
        r_rank=-np.log10(values["r_padj"].clip(lower=np.nextafter(0, 1))),
        py_rank=-np.log10(values["py_padj"].clip(lower=np.nextafter(0, 1))),
    )
    write_scatter(
        values,
        "r_rank",
        "py_rank",
        "R DESeq2 -log10(padj)",
        "PyDESeq2 -log10(padj)",
        f"{dataset_name} adjusted p-value scale concordance",
        outpath,
        inset_text=inset_text,
    )


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    timings = read_benchmark_timings(args.benchmark_timings, args.dataset_name, args.cpus)

    r_results = read_results(args.r_results)
    py_results = read_results(args.pydeseq2_results)
    r = r_results.add_prefix("r_").rename(columns={"r_gene_id": "gene_id"})
    py = py_results.add_prefix("py_").rename(columns={"py_gene_id": "gene_id"})
    joined = r.merge(py, on="gene_id", how="inner", validate="one_to_one")

    r_sig = significant_genes(r_results, "padj", args.alpha)
    py_sig = significant_genes(py_results, "padj", args.alpha)
    r_top = inclusive_top_ranked_genes(r_results, args.top_n)
    py_top = inclusive_top_ranked_genes(py_results, args.top_n)
    numeric_summary, na_disagreements, largest_differences = numeric_difference_diagnostics(
        r_results,
        py_results,
        joined,
    )
    sign_disagreements = sign_disagreement_table(joined)
    significance_disagreements = significance_disagreement_table(r_results, py_results, args.alpha)
    r_only_significant = significance_disagreements.loc[
        significance_disagreements["r_significant"] & ~significance_disagreements["pydeseq2_significant"]
    ]
    py_only_significant = significance_disagreements.loc[
        significance_disagreements["pydeseq2_significant"] & ~significance_disagreements["r_significant"]
    ]
    top_ranked = pd.concat(
        [
            r_top["table"].assign(implementation="r"),
            py_top["table"].assign(implementation="pydeseq2"),
        ],
        ignore_index=True,
    )
    top_ranked.insert(1, "requested_top_n", args.top_n)
    top_ranked = top_ranked[
        [
            "implementation",
            "requested_top_n",
            "gene_id",
            "padj",
            "pvalue",
            "rank_min",
            "rank_max",
            "boundary_member",
        ]
    ]

    signs = finite_pair(joined, "r_log2FoldChange", "py_log2FoldChange")
    nonzero = signs[(signs["r_log2FoldChange"] != 0) & (signs["py_log2FoldChange"] != 0)]
    sign_concordance = (
        (np.sign(nonzero["r_log2FoldChange"]) == np.sign(nonzero["py_log2FoldChange"])).mean()
        if len(nonzero) > 0
        else np.nan
    )
    sign_concordance_including_zero = (
        (np.sign(signs["r_log2FoldChange"]) == np.sign(signs["py_log2FoldChange"])).mean()
        if len(signs) > 0
        else np.nan
    )
    transform_summary = compare_transform_inputs(args, outdir)
    runtime_inputs = resolve_runtime_inputs(args, timings)
    r_core_runtime_seconds = runtime_inputs["r_core_runtime_seconds"]
    pydeseq2_core_runtime_seconds = runtime_inputs["pydeseq2_core_runtime_seconds"]
    pydeseq2_cores = runtime_inputs["pydeseq2_cores"]
    pydeseq2_vs_method = runtime_inputs["pydeseq2_vs_method"]
    r_nfcore_task_seconds = runtime_inputs["r_nfcore_task_seconds"]
    pydeseq2_nfcore_task_seconds = runtime_inputs["pydeseq2_nfcore_task_seconds"]
    repetitions = int(timings["repetition"].nunique()) if not timings.empty else None
    r_vst_runtime_seconds = timing_median(timings, "r", "vst_seconds")
    pydeseq2_vst_runtime_seconds = timing_median(timings, "pydeseq2", "vst_seconds")
    r_peak_rss_bytes = timing_value(timings, "r", "peak_rss_bytes", max)
    pydeseq2_peak_rss_bytes = timing_value(timings, "pydeseq2", "peak_rss_bytes", max)
    r_output_bytes = timing_median(timings, "r", "output_bytes")
    pydeseq2_output_bytes = timing_median(timings, "pydeseq2", "output_bytes")

    summary = {
        "dataset": args.dataset_name,
        "cpus": args.cpus,
        "repetitions": repetitions,
        "r_rows": int(len(r_results)),
        "pydeseq2_rows": int(len(py_results)),
        "shared_genes": int(len(joined)),
        "r_na_pvalue": int(r_results["pvalue"].isna().sum()) if "pvalue" in r_results else None,
        "r_na_padj": int(r_results["padj"].isna().sum()) if "padj" in r_results else None,
        "pydeseq2_na_pvalue": int(py_results["pvalue"].isna().sum()) if "pvalue" in py_results else None,
        "pydeseq2_na_padj": int(py_results["padj"].isna().sum()) if "padj" in py_results else None,
        "log2fc_pearson": correlation(joined, "r_log2FoldChange", "py_log2FoldChange", "pearson"),
        "log2fc_spearman": correlation(joined, "r_log2FoldChange", "py_log2FoldChange", "spearman"),
        "pvalue_spearman": correlation(joined, "r_pvalue", "py_pvalue", "spearman"),
        "padj_spearman": correlation(joined, "r_padj", "py_padj", "spearman"),
        "sign_concordance": sign_concordance,
        "sign_concordance_including_zero": sign_concordance_including_zero,
        "sign_pairs": int(len(signs)),
        "sign_nonzero_pairs": int(len(nonzero)),
        "r_significant": len(r_sig),
        "pydeseq2_significant": len(py_sig),
        "significant_overlap": len(r_sig & py_sig),
        "significant_jaccard": set_jaccard(r_sig, py_sig),
        "significance_disagreements": int(len(significance_disagreements)),
        "r_only_significant": int(len(r_only_significant)),
        "pydeseq2_only_significant": int(len(py_only_significant)),
        "sign_disagreements": int(len(sign_disagreements)),
        "na_mask_disagreements": int(len(na_disagreements)),
        "top_n": args.top_n,
        "top_n_overlap": len(r_top["genes"] & py_top["genes"]),
        "top_n_inclusive_overlap": len(r_top["genes"] & py_top["genes"]),
        "top_n_inclusive_jaccard": set_jaccard(r_top["genes"], py_top["genes"]),
        "r_top_n_eligible": r_top["eligible"],
        "pydeseq2_top_n_eligible": py_top["eligible"],
        "r_top_n_inclusive": r_top["included"],
        "pydeseq2_top_n_inclusive": py_top["included"],
        "r_top_n_boundary_padj": r_top["boundary_padj"],
        "r_top_n_boundary_pvalue": r_top["boundary_pvalue"],
        "pydeseq2_top_n_boundary_padj": py_top["boundary_padj"],
        "pydeseq2_top_n_boundary_pvalue": py_top["boundary_pvalue"],
        "r_top_n_boundary_tie_size": r_top["boundary_tie_size"],
        "pydeseq2_top_n_boundary_tie_size": py_top["boundary_tie_size"],
        "r_top_n_boundary_tied": r_top["boundary_tie_size"] > 1,
        "pydeseq2_top_n_boundary_tied": py_top["boundary_tie_size"] > 1,
        "r_top_n_boundary_tie_expansion": r_top["boundary_tie_expansion"],
        "pydeseq2_top_n_boundary_tie_expansion": py_top["boundary_tie_expansion"],
        "r_top_n_boundary_rank_min": r_top.get("boundary_rank_min"),
        "r_top_n_boundary_rank_max": r_top.get("boundary_rank_max"),
        "pydeseq2_top_n_boundary_rank_min": py_top.get("boundary_rank_min"),
        "pydeseq2_top_n_boundary_rank_max": py_top.get("boundary_rank_max"),
        "r_core_runtime_seconds": r_core_runtime_seconds,
        "r_core_runtime_source": runtime_inputs["r_core_runtime_source"],
        "pydeseq2_core_runtime_seconds": pydeseq2_core_runtime_seconds,
        "pydeseq2_core_runtime_source": runtime_inputs["pydeseq2_core_runtime_source"],
        "pydeseq2_cores": pydeseq2_cores,
        "pydeseq2_vs_method": pydeseq2_vs_method,
        "pydeseq2_over_r_core_runtime_ratio": runtime_ratio(
            pydeseq2_core_runtime_seconds,
            r_core_runtime_seconds,
        ),
        "r_nfcore_task_seconds": r_nfcore_task_seconds,
        "r_nfcore_task_source": runtime_inputs["r_nfcore_task_source"],
        "pydeseq2_nfcore_task_seconds": pydeseq2_nfcore_task_seconds,
        "pydeseq2_nfcore_task_source": runtime_inputs["pydeseq2_nfcore_task_source"],
        "pydeseq2_over_r_nfcore_task_ratio": runtime_ratio(
            pydeseq2_nfcore_task_seconds,
            r_nfcore_task_seconds,
        ),
        "r_vst_runtime_seconds": r_vst_runtime_seconds,
        "pydeseq2_vst_runtime_seconds": pydeseq2_vst_runtime_seconds,
        "r_peak_rss_bytes": r_peak_rss_bytes,
        "pydeseq2_peak_rss_bytes": pydeseq2_peak_rss_bytes,
        "r_output_bytes": r_output_bytes,
        "pydeseq2_output_bytes": pydeseq2_output_bytes,
        "r_results_deterministic": results_are_deterministic(timings, "r"),
        "pydeseq2_results_deterministic": results_are_deterministic(timings, "pydeseq2"),
        "alpha": args.alpha,
        "comparison_note": "Exploratory concordance check; not an equivalence test.",
        "ranking_note": (
            "The comparator does not round input values. It ranks by padj then pvalue and includes every gene "
            "tied at the requested boundary; gene identifiers never break biological-rank ties."
        ),
        "runtime_note": (
            "Core compares standalone R DESeq()+results() with PyDESeq2 module-recorded "
            "dds.deseq2()+DeseqStats.summary(); nf-core task seconds compare both module "
            "processes. Numeric CLI overrides are authoritative; the explicit R runtime file "
            "precedes benchmark medians, while the PyDESeq2 runtime TSV precedes legacy session "
            "timing; trace values are task-runtime fallbacks. Against the required benchmark table, external R "
            "core timing must match the selected CPU, dataset, dimensions, design, contrast, local version, "
            "and host provenance."
        ),
        **transform_summary,
    }

    for row in numeric_summary.itertuples(index=False):
        safe_column = re.sub(r"[^0-9A-Za-z]+", "_", row.column).strip("_").lower()
        summary[f"de_{safe_column}_abs_diff_median"] = row.abs_diff_median
        summary[f"de_{safe_column}_abs_diff_p95"] = row.abs_diff_p95
        summary[f"de_{safe_column}_abs_diff_p99"] = row.abs_diff_p99
        summary[f"de_{safe_column}_abs_diff_max"] = row.abs_diff_max
        summary[f"de_{safe_column}_na_mask_disagreements"] = row.na_mask_disagreements

    pd.DataFrame([summary]).to_csv(
        outdir / f"{args.output_prefix}_pydeseq2_vs_r_summary.tsv",
        sep="\t",
        index=False,
        na_rep="NA",
    )
    joined.to_csv(outdir / f"{args.output_prefix}_pydeseq2_vs_r_joined.tsv", sep="\t", index=False, na_rep="NA")
    numeric_summary.to_csv(
        outdir / f"{args.output_prefix}_numeric_difference_summary.tsv", sep="\t", index=False, na_rep="NA"
    )
    largest_differences.to_csv(
        outdir / f"{args.output_prefix}_largest_numeric_differences.tsv", sep="\t", index=False, na_rep="NA"
    )
    na_disagreements.to_csv(
        outdir / f"{args.output_prefix}_na_mask_disagreements.tsv", sep="\t", index=False, na_rep="NA"
    )
    sign_disagreements.to_csv(
        outdir / f"{args.output_prefix}_sign_disagreements.tsv", sep="\t", index=False, na_rep="NA"
    )
    significance_disagreements.to_csv(
        outdir / f"{args.output_prefix}_significance_disagreements.tsv", sep="\t", index=False, na_rep="NA"
    )
    r_only_significant.to_csv(
        outdir / f"{args.output_prefix}_r_only_significant.tsv", sep="\t", index=False, na_rep="NA"
    )
    py_only_significant.to_csv(
        outdir / f"{args.output_prefix}_pydeseq2_only_significant.tsv", sep="\t", index=False, na_rep="NA"
    )
    top_ranked.to_csv(
        outdir / f"{args.output_prefix}_top_ranked_inclusive.tsv", sep="\t", index=False, na_rep="NA"
    )
    if not timings.empty:
        timings.to_csv(outdir / f"{args.output_prefix}_runtime_runs.tsv", sep="\t", index=False)
    shared_label = stats_label(
        summary,
        [
            ("shared genes", "shared_genes", "int"),
            ("R rows", "r_rows", "int"),
            ("Py rows", "pydeseq2_rows", "int"),
            ("R task sec", "r_nfcore_task_seconds", "float"),
            ("Py task sec", "pydeseq2_nfcore_task_seconds", "float"),
            ("R core sec", "r_core_runtime_seconds", "float"),
            ("Py core sec", "pydeseq2_core_runtime_seconds", "float"),
        ],
    )
    lfc_label = shared_label + "\n" + stats_label(
        summary,
        [
            ("Pearson r", "log2fc_pearson", "float"),
            ("Spearman rho", "log2fc_spearman", "float"),
            ("sign concordance", "sign_concordance", "float"),
        ],
    )
    padj_label = shared_label + "\n" + stats_label(
        summary,
        [
            ("padj rho", "padj_spearman", "float"),
            ("R padj NA", "r_na_padj", "int"),
            ("Py padj NA", "pydeseq2_na_padj", "int"),
            ("sig overlap", "significant_overlap", "int"),
        ],
    )
    rank_label = shared_label + "\n" + stats_label(
        summary,
        [
            ("requested N", "top_n", "int"),
            ("inclusive overlap", "top_n_inclusive_overlap", "int"),
            ("R inclusive", "r_top_n_inclusive", "int"),
            ("Py inclusive", "pydeseq2_top_n_inclusive", "int"),
            ("pvalue rho", "pvalue_spearman", "float"),
            ("Py/R task", "pydeseq2_over_r_nfcore_task_ratio", "float"),
            ("Py/R core", "pydeseq2_over_r_core_runtime_ratio", "float"),
        ],
    )
    write_scatter(
        joined,
        "r_log2FoldChange",
        "py_log2FoldChange",
        "R DESeq2 log2FoldChange",
        "PyDESeq2 log2FoldChange",
        f"{args.dataset_name} log2 fold-change concordance",
        outdir / f"{args.output_prefix}_log2fc_scatter.png",
        inset_text=lfc_label,
    )
    write_scatter(
        joined,
        "r_padj",
        "py_padj",
        "R DESeq2 padj",
        "PyDESeq2 padj",
        f"{args.dataset_name} adjusted p-value concordance",
        outdir / f"{args.output_prefix}_padj_scatter.png",
        inset_text=padj_label,
    )
    write_rank_plot(
        joined,
        args.dataset_name,
        outdir / f"{args.output_prefix}_ranked_neglog10_padj_scatter.png",
        rank_label,
    )
    with open(outdir / f"{args.output_prefix}_pydeseq2_vs_r_summary.json", "w", encoding="utf-8") as handle:
        json.dump({key: json_value(value) for key, value in summary.items()}, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
