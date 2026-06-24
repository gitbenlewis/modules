#!/usr/bin/env python

import json
import platform
import re
import shlex
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats


ARGS = r'''${args}'''

DEFAULTS = {
    "output_prefix": "${prefix}",
    "count_file": "${counts}",
    "sample_file": "${samplesheet}",
    "contrast_variable": "${contrast_variable}",
    "reference_level": "${reference}",
    "target_level": "${target}",
    "contrast_string": "${comparison}",
    "formula": "${formula}",
    "blocking_variables": None,
    "control_genes_file": "${control_genes_file}",
    "transcript_lengths_file": "${transcript_lengths_file}",
    "sizefactors_from_controls": False,
    "gene_id_col": "gene_id",
    "sample_id_col": "experiment_accession",
    "subset_to_contrast_samples": False,
    "exclude_samples_col": None,
    "exclude_samples_values": None,
    "test": "Wald",
    "fit_type": "parametric",
    "sf_type": "ratio",
    "min_replicates_for_replace": 7,
    "use_t": False,
    "lfc_threshold": 0.0,
    "alt_hypothesis": "greaterAbs",
    "independent_filtering": True,
    "p_adjust_method": "BH",
    "alpha": 0.1,
    "minmu": 0.5,
    "vs_method": "vst",
    "shrink_lfc": False,
    "cores": 1,
    "vs_blind": True,
    "vst_nsub": 1000,
    "round_digits": None,
    "seed": None,
    "feature_filtering": True,
    "filtering_min_samples": 1,
    "filtering_min_abundance": 1.0,
    "dispersion_fallback_policy": "warn",
}

TYPE_OVERRIDES = {
    "round_digits": int,
    "seed": int,
    "filtering_min_samples": int,
    "filtering_min_abundance": float,
    "blocking_variables": str,
    "exclude_samples_col": str,
    "exclude_samples_values": str,
}


def valid_string(value):
    return value is not None and str(value).strip() not in {"", "null", "None", "[]"}


def optional_file(value):
    return valid_string(value) and Path(str(value)).exists()


def parse_bool(value):
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes"}:
        return True
    if text in {"false", "f", "0", "no"}:
        return False
    raise ValueError(f"Expected a boolean value, got '{value}'")


def nullify(value):
    if value is None:
        return None
    text = str(value).strip()
    return None if text in {"", "null", "None", "[]"} else value


def coerce_option(key, value):
    default = DEFAULTS[key]
    target_type = TYPE_OVERRIDES.get(key, type(default))
    if value is None or str(value).strip() in {"null", "None", ""}:
        return None
    if target_type is bool:
        return parse_bool(value)
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return str(value)


def parse_ext_args(args):
    tokens = shlex.split(args or "")
    parsed = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise ValueError(f"Invalid option token '{token}'")
        key = token[2:].replace("-", "_")
        if key not in DEFAULTS:
            raise ValueError(f"Invalid option: {key}")
        if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
            index += 1
            continue
        value = tokens[index + 1]
        index += 2
        parsed[key] = coerce_option(key, value)
    return parsed


def make_name(value):
    text = re.sub(r"[^0-9A-Za-z_.]", ".", str(value))
    text = re.sub(r"^[^A-Za-z_.]", lambda match: "X" + match.group(0), text)
    return text


def make_unique(values):
    seen = {}
    names = []
    for value in values:
        name = make_name(value)
        count = seen.get(name, 0)
        seen[name] = count + 1
        names.append(name if count == 0 else f"{name}.{count}")
    return names


def read_delim_flexible(path, **kwargs):
    suffix = Path(path).suffix.lower()
    if suffix in {".tsv", ".txt"}:
        sep = "\t"
    elif suffix == ".csv":
        sep = ","
    else:
        raise ValueError(f"Unknown separator for '{path}'")
    return pd.read_csv(path, sep=sep, **kwargs)


def unsupported(message):
    raise ValueError(f"Unsupported PyDESeq2 module option: {message}")


def validate_options(opt):
    if optional_file(opt["transcript_lengths_file"]):
        unsupported(
            "transcript length correction is not available through PyDESeq2's public API. "
            "R DESeq2 stores avgTxLength and derives gene-by-sample normalization factors; "
            "this module fails instead of silently ignoring transcript lengths."
        )
    if str(opt["test"]) != "Wald":
        unsupported("only Wald tests are supported")
    if opt["use_t"]:
        unsupported("use_t is an R DESeq2 option without PyDESeq2 parity")
    if str(opt["p_adjust_method"]).upper() != "BH":
        unsupported("only BH p-value adjustment is supported")
    if opt["fit_type"] not in {"parametric", "mean"}:
        unsupported("fit_type must be 'parametric' or 'mean'")
    if opt["dispersion_fallback_policy"] not in {"warn", "error"}:
        unsupported("dispersion_fallback_policy must be 'warn' or 'error'")
    if opt["sf_type"] == "iterate":
        opt["sf_type"] = "iterative"
    if opt["sf_type"] not in {"ratio", "poscounts", "iterative"}:
        unsupported("sf_type must be 'ratio', 'poscounts', or 'iterative'")
    if opt["vst_nsub"] != DEFAULTS["vst_nsub"]:
        unsupported("vst_nsub is not exposed by PyDESeq2 VST")
    if opt["shrink_lfc"]:
        unsupported("ASHR-compatible shrink_lfc is not available; set --shrink_lfc FALSE")
    if opt["filtering_min_samples"] < 1:
        raise ValueError("filtering_min_samples must be at least 1; set feature_filtering false to disable filtering")
    if opt["filtering_min_abundance"] < 0:
        raise ValueError("filtering_min_abundance must be non-negative")
    if any(method.strip() == "rlog" for method in str(opt["vs_method"]).split(",")):
        unsupported("rlog is an R DESeq2 transform without PyDESeq2 parity")
    if any(method.strip() not in {"", "vst"} for method in str(opt["vs_method"]).split(",")):
        unsupported("vs_method supports only 'vst'")


def split_semicolon(value):
    if not valid_string(value):
        return []
    return [item for item in str(value).split(";") if item]


def round_numeric(df, digits):
    if digits is None:
        return df
    rounded = df.copy()
    numeric_cols = rounded.select_dtypes(include=[np.number]).columns
    rounded[numeric_cols] = rounded[numeric_cols].round(digits)
    return rounded


def write_tsv(df, path):
    df.to_csv(path, sep="\t", index=False, na_rep="NA")


def matrix_to_output(matrix, genes, samples, gene_id_col, round_digits):
    df = pd.DataFrame(matrix, index=samples, columns=genes).T
    df = round_numeric(df, round_digits)
    df.insert(0, gene_id_col, df.index)
    return df.reset_index(drop=True)


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def write_versions():
    versions = {
        "python": platform.python_version(),
        "pydeseq2": package_version("pydeseq2"),
        "matplotlib": matplotlib.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    with open("versions.yml", "w", encoding="utf-8") as handle:
        handle.write('"${task.process}":\\n')
        for key, value in versions.items():
            handle.write(f"    {key}: {value}\\n")


def jsonable(value):
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def record_dispersion_fit(dds, opt):
    actual_fit_type = dds.uns.get("disp_function_type")
    opt["pydeseq2_requested_dispersion_fit_type"] = opt["fit_type"]
    opt["pydeseq2_actual_dispersion_fit_type"] = actual_fit_type
    opt["pydeseq2_dispersion_fit_fallback"] = opt["fit_type"] != actual_fit_type
    opt["pydeseq2_trend_coeffs"] = jsonable(dds.uns.get("trend_coeffs"))
    opt["pydeseq2_mean_dispersion"] = jsonable(dds.uns.get("mean_disp"))
    opt["pydeseq2_prior_dispersion_variance"] = jsonable(dds.uns.get("prior_disp_var"))

    if opt["pydeseq2_dispersion_fit_fallback"]:
        message = (
            f"PyDESeq2 requested dispersion fit_type='{opt['fit_type']}' but used "
            f"'{actual_fit_type}'."
        )
        if opt["dispersion_fallback_policy"] == "error":
            raise RuntimeError(message + " Set --dispersion_fallback_policy warn to allow this fallback.")
        print("WARNING: " + message, file=sys.stderr)


def prepare_inputs(opt):
    count_table = read_delim_flexible(opt["count_file"])
    if opt["gene_id_col"] not in count_table.columns:
        raise ValueError(f"Specified gene ID column '{opt['gene_id_col']}' is not in the counts file")
    count_table = count_table.set_index(opt["gene_id_col"])

    sample_sheet = read_delim_flexible(opt["sample_file"])
    sample_sheet.columns = make_unique(sample_sheet.columns)

    sample_id_col = make_name(opt["sample_id_col"])
    if sample_id_col not in sample_sheet.columns:
        raise ValueError(f"Specified sample ID column '{sample_id_col}' is not in the sample sheet")

    sample_sheet = sample_sheet.loc[~sample_sheet[sample_id_col].duplicated()].copy()
    sample_sheet.index = sample_sheet[sample_id_col].astype(str)

    missing_samples = sample_sheet.index.difference(count_table.columns)
    if len(missing_samples) > 0:
        missing = ",".join(missing_samples.astype(str))
        raise ValueError(f"{len(missing_samples)} specified samples missing from count table: {missing}")
    count_table = count_table.loc[:, sample_sheet.index]
    return count_table, sample_sheet, sample_id_col


def apply_sample_filters(count_table, sample_sheet, sample_id_col, opt, contrast_variable):
    if contrast_variable is not None and not valid_string(opt["formula"]) and opt["subset_to_contrast_samples"]:
        selector = sample_sheet[contrast_variable].isin([opt["target_level"], opt["reference_level"]])
        selected = sample_sheet.loc[selector, sample_id_col].astype(str)
        count_table = count_table.loc[:, selected]
        sample_sheet = sample_sheet.loc[selected].copy()

    if valid_string(opt["exclude_samples_col"]) and valid_string(opt["exclude_samples_values"]):
        exclude_col = make_name(opt["exclude_samples_col"])
        if exclude_col not in sample_sheet.columns:
            raise ValueError(f"{exclude_col} specified to subset samples is not a valid sample sheet column")
        exclude_values = set(split_semicolon(opt["exclude_samples_values"]))
        selector = ~sample_sheet[exclude_col].astype(str).isin(exclude_values)
        selected = sample_sheet.loc[selector, sample_id_col].astype(str)
        count_table = count_table.loc[:, selected]
        sample_sheet = sample_sheet.loc[selected].copy()
    return count_table, sample_sheet


def apply_feature_filter(count_table, opt):
    features_before = len(count_table)
    if not opt["feature_filtering"]:
        return count_table, features_before, features_before

    keep = (count_table >= opt["filtering_min_abundance"]).sum(axis=1) >= opt["filtering_min_samples"]
    filtered = count_table.loc[keep].copy()
    if filtered.empty:
        raise ValueError(
            "Feature filtering removed all genes. "
            "Lower filtering_min_samples/filtering_min_abundance or set feature_filtering false."
        )
    return filtered, features_before, len(filtered)


def build_model_and_contrast(opt, sample_sheet):
    if valid_string(opt["formula"]):
        if not valid_string(opt["contrast_string"]):
            raise ValueError("formula requires a comparison contrast string")
        return str(opt["formula"]), None

    contrast_variable = make_name(opt["contrast_variable"])
    if contrast_variable not in sample_sheet.columns:
        raise ValueError(f"Chosen contrast variable '{contrast_variable}' not in sample sheet")
    if not {opt["reference_level"], opt["target_level"]}.issubset(set(sample_sheet[contrast_variable].astype(str))):
        raise ValueError("Please choose reference and treatment levels present in the contrast column")

    blocking_vars = [make_name(value) for value in split_semicolon(opt["blocking_variables"])]
    missing_blocking = [value for value in blocking_vars if value not in sample_sheet.columns]
    if missing_blocking:
        raise ValueError(f"Blocking variables {','.join(missing_blocking)} do not correspond to sample sheet columns")

    for column in [*blocking_vars, contrast_variable]:
        sample_sheet[column] = sample_sheet[column].astype("category")

    terms = [*blocking_vars, contrast_variable]
    model = "~ " + " + ".join(terms)
    return model, [contrast_variable, str(opt["target_level"]), str(opt["reference_level"])]


def formula_contrast_vector(dds, comparison):
    columns = list(dds.obsm["design_matrix"].columns)
    if comparison in columns:
        contrast = np.zeros(len(columns))
        contrast[columns.index(comparison)] = 1.0
        return contrast
    if "-" in comparison:
        positive, negative = [item.strip() for item in comparison.split("-", 1)]
        if positive in columns and negative in columns:
            contrast = np.zeros(len(columns))
            contrast[columns.index(positive)] = 1.0
            contrast[columns.index(negative)] = -1.0
            return contrast
    raise ValueError(
        "Formula comparison must match a PyDESeq2 design-matrix coefficient. "
        f"Got '{comparison}'. Available coefficients: {', '.join(columns)}"
    )


def write_session_info(opt):
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pydeseq2": package_version("pydeseq2"),
        "matplotlib": matplotlib.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "options": opt,
    }
    with open(f"{opt['output_prefix']}.Python_sessionInfo.log", "w", encoding="utf-8") as handle:
        handle.write("PyDESeq2 session information\\n")
        handle.write(json.dumps(payload, indent=2, sort_keys=True, default=str))
        handle.write("\\n")


def main():
    opt = DEFAULTS.copy()
    opt.update(parse_ext_args(ARGS))
    for key in ("formula", "contrast_string", "contrast_variable", "reference_level", "target_level", "seed"):
        opt[key] = nullify(opt[key])
    validate_options(opt)

    if opt["seed"] is not None:
        np.random.seed(opt["seed"])

    count_table, sample_sheet, sample_id_col = prepare_inputs(opt)
    count_table, features_before_filter, features_after_filter = apply_feature_filter(count_table, opt)
    opt["features_before_filter"] = features_before_filter
    opt["features_after_filter"] = features_after_filter
    contrast_variable = make_name(opt["contrast_variable"]) if valid_string(opt["contrast_variable"]) else None
    count_table, sample_sheet = apply_sample_filters(count_table, sample_sheet, sample_id_col, opt, contrast_variable)

    model, contrast = build_model_and_contrast(opt, sample_sheet)

    control_genes = None
    if optional_file(opt["control_genes_file"]):
        control_genes = [line.strip() for line in Path(opt["control_genes_file"]).read_text().splitlines() if line.strip()]
        if opt["sizefactors_from_controls"]:
            control_genes = [gene for gene in control_genes if gene in count_table.index]
            if not control_genes:
                raise ValueError("No supplied control genes were present in the count table")
        else:
            count_table = count_table.loc[~count_table.index.isin(control_genes)]
            control_genes = None

    counts_for_pydeseq2 = count_table.round().astype(int).T
    inference = DefaultInference(n_cpus=opt["cores"])
    dds = DeseqDataSet(
        counts=counts_for_pydeseq2,
        metadata=sample_sheet,
        design=model,
        fit_type=opt["fit_type"],
        size_factors_fit_type=opt["sf_type"],
        control_genes=control_genes,
        min_mu=opt["minmu"],
        refit_cooks=True,
        min_replicates=opt["min_replicates_for_replace"],
        inference=inference,
        quiet=True,
    )
    dds.deseq2()
    record_dispersion_fit(dds, opt)

    if contrast is None:
        contrast = formula_contrast_vector(dds, str(opt["contrast_string"]))

    stats = DeseqStats(
        dds,
        contrast=contrast,
        alpha=opt["alpha"],
        independent_filter=opt["independent_filtering"],
        lfc_null=opt["lfc_threshold"],
        alt_hypothesis=opt["alt_hypothesis"],
        inference=inference,
        quiet=True,
        n_cpus=opt["cores"],
    )
    stats.summary()

    prefix = opt["output_prefix"]
    results = round_numeric(stats.results_df.copy(), opt["round_digits"])
    results.insert(0, opt["gene_id_col"], results.index)
    write_tsv(results.reset_index(drop=True), f"{prefix}.pydeseq2.results.tsv")

    dds.plot_dispersions(save_path=f"{prefix}.pydeseq2.dispersion.png")
    plt.close("all")
    dds.plot_dispersions(save_path=f"{prefix}.pydeseq2.dispersion.pdf")
    plt.close("all")

    adata = dds.to_picklable_anndata()
    if isinstance(adata.uns.get("trend_coeffs"), pd.Series):
        adata.uns["trend_coeffs"] = adata.uns["trend_coeffs"].to_dict()
    adata.write_h5ad(f"{prefix}.pydeseq2.dds.h5ad")

    size_factors = pd.DataFrame({"sample": dds.obs_names, "sizeFactor": dds.obs["size_factors"].to_numpy()})
    write_tsv(size_factors, f"{prefix}.pydeseq2.sizefactors.tsv")

    normalised = matrix_to_output(
        dds.layers["normed_counts"],
        dds.var_names,
        dds.obs_names,
        opt["gene_id_col"],
        opt["round_digits"],
    )
    write_tsv(normalised, f"{prefix}.normalised_counts.tsv")

    if "vst" in {method.strip() for method in str(opt["vs_method"]).split(",")}:
        dds.vst(use_design=not opt["vs_blind"], fit_type=opt["fit_type"])
        vst = matrix_to_output(
            dds.layers["vst_counts"],
            dds.var_names,
            dds.obs_names,
            opt["gene_id_col"],
            opt["round_digits"],
        )
        write_tsv(vst, f"{prefix}.vst.tsv")

    Path(f"{prefix}.pydeseq2.model.txt").write_text(model + "\\n", encoding="utf-8")
    write_session_info(opt)
    write_versions()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
