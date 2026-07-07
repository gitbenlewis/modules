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


def parse_args():
    parser = argparse.ArgumentParser(description="Compare pasilla R DESeq2 and PyDESeq2 result tables.")
    parser.add_argument("--r-results", required=True, help="R DESeq2 results TSV with gene_id column.")
    parser.add_argument("--pydeseq2-results", required=True, help="PyDESeq2 results TSV with gene_id column.")
    parser.add_argument("--outdir", default="work/pasilla_compare", help="Output directory.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Adjusted p-value significance threshold.")
    parser.add_argument("--top-n", type=int, default=100, help="Top-N genes to compare by adjusted p-value rank.")
    parser.add_argument("--r-runtime", help="Optional R runtime TSV written by prepare_pasilla_reference.R.")
    parser.add_argument("--pydeseq2-trace", help="Optional Nextflow trace file from the PyDESeq2 example run.")
    parser.add_argument("--pydeseq2-runtime-seconds", type=float, help="Optional PyDESeq2 elapsed seconds override.")
    return parser.parse_args()


def read_results(path):
    frame = pd.read_csv(path, sep="\t", na_values=["NA", "NaN", ""], keep_default_na=True)
    if "gene_id" not in frame.columns:
        raise ValueError(f"{path} does not contain a gene_id column")
    return frame


def finite_pair(frame, left, right):
    values = frame[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
    return values


def correlation(frame, left, right, method):
    values = finite_pair(frame, left, right)
    if len(values) < 2:
        return np.nan
    return values[left].corr(values[right], method=method)


def significant_genes(frame, column, alpha):
    if column not in frame:
        return set()
    values = frame[["gene_id", column]].dropna()
    return set(values.loc[values[column] < alpha, "gene_id"])


def top_genes(frame, column, n):
    if column not in frame:
        return set()
    values = frame[["gene_id", column]].dropna().sort_values([column, "gene_id"])
    return set(values.head(n)["gene_id"])


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
    for amount, unit in re.findall(r"([0-9]*\.?[0-9]+)\s*([dhms])", text):
        matched = True
        scale = {"d": 86400, "h": 3600, "m": 60, "s": 1}[unit]
        total += float(amount) * scale
    return total if matched else np.nan


def read_r_runtime(path):
    if not path:
        return None
    runtime = pd.read_csv(path, sep="\t")
    if {"step", "elapsed_seconds"}.issubset(runtime.columns):
        selector = runtime["step"] == "deseq_and_results"
        if selector.any():
            return float(runtime.loc[selector, "elapsed_seconds"].iloc[0])
        if len(runtime) > 0:
            return float(runtime["elapsed_seconds"].iloc[0])
    return None


def read_pydeseq2_runtime(trace_path, override):
    if override is not None:
        return float(override)
    if not trace_path:
        return None
    trace = pd.read_csv(trace_path, sep="\t")
    if "process" in trace.columns:
        trace = trace.loc[trace["process"].astype(str).str.contains("PYDESEQ2_DIFFERENTIAL", na=False)]
    for column in ("realtime", "duration"):
        if column in trace.columns and len(trace) > 0:
            values = trace[column].map(parse_duration_seconds).dropna()
            if len(values) > 0:
                return float(values.sum())
    return None


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


def write_rank_plot(frame, outpath, inset_text):
    values = frame[["gene_id", "r_padj", "py_padj"]].replace([np.inf, -np.inf], np.nan).dropna()
    values = values.assign(
        r_rank=-np.log10(values["r_padj"].clip(lower=np.nextafter(0, 1))),
        py_rank=-np.log10(values["py_padj"].clip(lower=np.nextafter(0, 1))),
    )
    write_scatter(
        values,
        "r_rank",
        "py_rank",
        "R DESeq2 ranked -log10(padj)",
        "PyDESeq2 ranked -log10(padj)",
        "Pasilla ranked adjusted p-value concordance",
        outpath,
        inset_text=inset_text,
    )


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    r = read_results(args.r_results).add_prefix("r_").rename(columns={"r_gene_id": "gene_id"})
    py = read_results(args.pydeseq2_results).add_prefix("py_").rename(columns={"py_gene_id": "gene_id"})
    joined = r.merge(py, on="gene_id", how="inner")

    r_sig = significant_genes(r.rename(columns={"r_padj": "padj"}), "padj", args.alpha)
    py_sig = significant_genes(py.rename(columns={"py_padj": "padj"}), "padj", args.alpha)
    r_top = top_genes(r.rename(columns={"r_padj": "padj"}), "padj", args.top_n)
    py_top = top_genes(py.rename(columns={"py_padj": "padj"}), "padj", args.top_n)

    signs = finite_pair(joined, "r_log2FoldChange", "py_log2FoldChange")
    nonzero = signs[(signs["r_log2FoldChange"] != 0) & (signs["py_log2FoldChange"] != 0)]
    sign_concordance = (
        (np.sign(nonzero["r_log2FoldChange"]) == np.sign(nonzero["py_log2FoldChange"])).mean()
        if len(nonzero) > 0
        else np.nan
    )
    r_runtime_seconds = read_r_runtime(args.r_runtime)
    pydeseq2_runtime_seconds = read_pydeseq2_runtime(args.pydeseq2_trace, args.pydeseq2_runtime_seconds)
    runtime_ratio = (
        pydeseq2_runtime_seconds / r_runtime_seconds
        if r_runtime_seconds not in (None, 0) and pydeseq2_runtime_seconds is not None
        else None
    )

    summary = {
        "r_rows": int(len(r)),
        "pydeseq2_rows": int(len(py)),
        "shared_genes": int(len(joined)),
        "r_na_pvalue": int(r["r_pvalue"].isna().sum()) if "r_pvalue" in r else None,
        "r_na_padj": int(r["r_padj"].isna().sum()) if "r_padj" in r else None,
        "pydeseq2_na_pvalue": int(py["py_pvalue"].isna().sum()) if "py_pvalue" in py else None,
        "pydeseq2_na_padj": int(py["py_padj"].isna().sum()) if "py_padj" in py else None,
        "log2fc_pearson": correlation(joined, "r_log2FoldChange", "py_log2FoldChange", "pearson"),
        "log2fc_spearman": correlation(joined, "r_log2FoldChange", "py_log2FoldChange", "spearman"),
        "pvalue_spearman": correlation(joined, "r_pvalue", "py_pvalue", "spearman"),
        "padj_spearman": correlation(joined, "r_padj", "py_padj", "spearman"),
        "sign_concordance": sign_concordance,
        "r_significant": len(r_sig),
        "pydeseq2_significant": len(py_sig),
        "significant_overlap": len(r_sig & py_sig),
        "top_n": args.top_n,
        "top_n_overlap": len(r_top & py_top),
        "r_runtime_seconds": r_runtime_seconds,
        "pydeseq2_runtime_seconds": pydeseq2_runtime_seconds,
        "pydeseq2_over_r_runtime_ratio": runtime_ratio,
        "alpha": args.alpha,
        "comparison_note": "Exploratory concordance check; not an equivalence test.",
    }

    pd.DataFrame([summary]).to_csv(outdir / "pasilla_pydeseq2_vs_r_summary.tsv", sep="\t", index=False)
    joined.to_csv(outdir / "pasilla_pydeseq2_vs_r_joined.tsv", sep="\t", index=False, na_rep="NA")
    shared_label = stats_label(
        summary,
        [
            ("shared genes", "shared_genes", "int"),
            ("R rows", "r_rows", "int"),
            ("Py rows", "pydeseq2_rows", "int"),
            ("R sec", "r_runtime_seconds", "float"),
            ("Py sec", "pydeseq2_runtime_seconds", "float"),
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
            ("top N", "top_n", "int"),
            ("top overlap", "top_n_overlap", "int"),
            ("pvalue rho", "pvalue_spearman", "float"),
            ("Py/R runtime", "pydeseq2_over_r_runtime_ratio", "float"),
        ],
    )
    write_scatter(
        joined,
        "r_log2FoldChange",
        "py_log2FoldChange",
        "R DESeq2 log2FoldChange",
        "PyDESeq2 log2FoldChange",
        "Pasilla log2 fold-change concordance",
        outdir / "pasilla_log2fc_scatter.png",
        inset_text=lfc_label,
    )
    write_scatter(
        joined,
        "r_padj",
        "py_padj",
        "R DESeq2 padj",
        "PyDESeq2 padj",
        "Pasilla adjusted p-value concordance",
        outdir / "pasilla_padj_scatter.png",
        inset_text=padj_label,
    )
    write_rank_plot(joined, outdir / "pasilla_ranked_neglog10_padj_scatter.png", rank_label)
    with open(outdir / "pasilla_pydeseq2_vs_r_summary.json", "w", encoding="utf-8") as handle:
        json.dump({key: json_value(value) for key, value in summary.items()}, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
