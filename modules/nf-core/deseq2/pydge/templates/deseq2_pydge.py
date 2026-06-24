#!/usr/bin/env python

import json
import platform
import re
import shlex
import sys
import types
import warnings
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend
from pydeseq2 import utils
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from pydeseq2.preprocessing import deseq2_norm_fit
from scipy.stats import f


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
    "r_style_parametric_dispersion_fit": True,
    "parametric_dispersion_min_disp_multiplier": 100.0,
    "transcript_length_normalization": True,
    "transcript_length_normalization_policy": "error",
}

TYPE_OVERRIDES = {
    "round_digits": int,
    "seed": int,
    "filtering_min_samples": int,
    "filtering_min_abundance": float,
    "parametric_dispersion_min_disp_multiplier": float,
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
    if opt["transcript_length_normalization_policy"] != "error":
        unsupported("transcript_length_normalization_policy currently supports only 'error'")
    if optional_file(opt["transcript_lengths_file"]) and not opt["transcript_length_normalization"]:
        unsupported("transcript lengths were supplied but transcript_length_normalization is false")
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
    if opt["parametric_dispersion_min_disp_multiplier"] < 0:
        raise ValueError("parametric_dispersion_min_disp_multiplier must be non-negative")
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


class NormalizationFactorInference(DefaultInference):
    def lin_reg_mu(self, counts, size_factors, design_matrix, min_mu):
        if np.ndim(size_factors) != 2:
            return super().lin_reg_mu(counts, size_factors, design_matrix, min_mu)

        with parallel_backend(self._backend, inner_max_num_threads=1):
            mu_hat_ = np.array(
                Parallel(
                    n_jobs=self.n_cpus,
                    verbose=self._joblib_verbosity,
                    batch_size=self._batch_size,
                )(
                    delayed(utils.fit_lin_mu)(
                        counts=counts[:, i],
                        size_factors=size_factors[:, i],
                        design_matrix=design_matrix,
                        min_mu=min_mu,
                    )
                    for i in range(counts.shape[1])
                )
            )
        return mu_hat_.T

    def irls(
        self,
        counts,
        size_factors,
        design_matrix,
        disp,
        min_mu,
        beta_tol,
        min_beta=-30,
        max_beta=30,
        optimizer="L-BFGS-B",
        maxiter=250,
    ):
        if np.ndim(size_factors) != 2:
            return super().irls(
                counts,
                size_factors,
                design_matrix,
                disp,
                min_mu,
                beta_tol,
                min_beta,
                max_beta,
                optimizer,
                maxiter,
            )

        with parallel_backend(self._backend, inner_max_num_threads=1):
            res = Parallel(
                n_jobs=self.n_cpus,
                verbose=self._joblib_verbosity,
                batch_size=self._batch_size,
            )(
                delayed(utils.irls_solver)(
                    counts=counts[:, i],
                    size_factors=size_factors[:, i],
                    design_matrix=design_matrix,
                    disp=disp[i],
                    min_mu=min_mu,
                    beta_tol=beta_tol,
                    min_beta=min_beta,
                    max_beta=max_beta,
                    optimizer=optimizer,
                    maxiter=maxiter,
                )
                for i in range(counts.shape[1])
            )
        res = zip(*res, strict=False)
        mle_lfcs, mu_hat, hat_diagonals, converged = (np.array(m) for m in res)
        return mle_lfcs, mu_hat.T, hat_diagonals.T, converged


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


def read_transcript_lengths(opt, count_table):
    if not optional_file(opt["transcript_lengths_file"]):
        opt["transcript_length_normalization_enabled"] = False
        opt["transcript_length_file_used"] = None
        return None

    lengths = read_delim_flexible(opt["transcript_lengths_file"])
    if opt["gene_id_col"] not in lengths.columns:
        raise ValueError(f"Specified gene ID column '{opt['gene_id_col']}' is not in the transcript length file")
    lengths = lengths.set_index(opt["gene_id_col"])

    missing_genes = count_table.index.difference(lengths.index)
    missing_samples = count_table.columns.difference(lengths.columns)
    if len(missing_genes) > 0 or len(missing_samples) > 0:
        raise ValueError(
            "Transcript length matrix is incompatible with the retained count matrix: "
            f"{len(missing_genes)} missing genes and {len(missing_samples)} missing samples"
        )

    lengths = lengths.loc[count_table.index, count_table.columns].apply(pd.to_numeric, errors="coerce")
    values = lengths.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Transcript length matrix must contain finite positive values for all retained genes and samples")

    opt["transcript_length_normalization_enabled"] = True
    opt["transcript_length_file_used"] = str(opt["transcript_lengths_file"])
    opt["transcript_length_matrix_shape"] = [int(lengths.shape[0]), int(lengths.shape[1])]
    return lengths


def build_normalization_factors(count_table, transcript_lengths, opt):
    if transcript_lengths is None:
        return None, None

    length_values = transcript_lengths.to_numpy(dtype=float)
    with np.errstate(divide="raise", invalid="raise"):
        gene_geo_means = np.exp(np.log(length_values).mean(axis=1))
    length_norm = pd.DataFrame(
        length_values / gene_geo_means[:, None],
        index=transcript_lengths.index,
        columns=transcript_lengths.columns,
    )
    adjusted_counts = count_table / length_norm
    adjusted_by_sample = adjusted_counts.T
    logmeans, filtered_genes = deseq2_norm_fit(adjusted_by_sample)
    logmeans = np.asarray(logmeans)
    filtered_genes = np.asarray(filtered_genes)
    if not np.any(filtered_genes):
        raise ValueError("No genes were usable for transcript-length adjusted size-factor estimation")

    adjusted_matrix = adjusted_by_sample.to_numpy(dtype=float)
    with np.errstate(divide="ignore"):
        log_counts = np.log(adjusted_matrix)
    log_ratios = log_counts[:, filtered_genes] - logmeans[filtered_genes]
    sample_size_factors = pd.Series(
        np.exp(np.median(log_ratios, axis=1)),
        index=adjusted_by_sample.index,
        name="sizeFactor",
    )
    if not np.isfinite(sample_size_factors.to_numpy()).all() or (sample_size_factors <= 0).any():
        raise ValueError("Transcript-length adjusted size-factor estimation produced non-positive or non-finite values")

    normalization_factors = length_norm.mul(sample_size_factors, axis=1)
    opt["transcript_length_adjusted_size_factor_genes"] = int(np.sum(filtered_genes))
    opt["normalization_factor_shape"] = [int(normalization_factors.shape[1]), int(normalization_factors.shape[0])]
    opt["sample_size_factor_summary"] = {
        "min": float(sample_size_factors.min()),
        "median": float(sample_size_factors.median()),
        "max": float(sample_size_factors.max()),
    }
    opt["transcript_length_cooks_refit_used_normalization_factors"] = False
    return normalization_factors.T, sample_size_factors


def install_transcript_length_normalization(dds, normalization_factors, sample_size_factors, opt):
    if normalization_factors is None:
        return

    normalization_factors = normalization_factors.loc[dds.obs_names, dds.var_names]
    sample_size_factors = sample_size_factors.loc[dds.obs_names]
    dds.obsm["normalization_factors"] = normalization_factors.to_numpy(dtype=float)
    dds.obs["size_factors"] = sample_size_factors.to_numpy(dtype=float)
    dds.layers["normed_counts"] = dds.X / dds.obsm["normalization_factors"]
    dds.var["_normed_means"] = dds.layers["normed_counts"].mean(axis=0)

    def fit_size_factors_with_normalization_factors(self, fit_type=None, control_genes=None):
        self.obs["size_factors"] = sample_size_factors.loc[self.obs_names].to_numpy(dtype=float)
        self.obsm["normalization_factors"] = normalization_factors.loc[self.obs_names, self.var_names].to_numpy(dtype=float)
        self.layers["normed_counts"] = self.X / self.obsm["normalization_factors"]
        self.var["_normed_means"] = self.layers["normed_counts"].mean(axis=0)
        self.logmeans = np.log(self.layers["normed_counts"]).mean(axis=0)
        self.filtered_genes = np.ones(self.n_vars, dtype=bool)

    def fit_moments_with_normalization_factors(self):
        if "normed_counts" not in self.layers:
            self.fit_size_factors(fit_type=self.size_factors_fit_type)

        normed_counts = self.layers["normed_counts"][:, self.non_zero_idx]
        norm_factors = self.obsm["normalization_factors"][:, self.non_zero_idx]
        rde = self.inference.fit_rough_dispersions(normed_counts, self.obsm["design_matrix"].values)
        mde = self.inference.fit_moments_dispersions(normed_counts, norm_factors)
        alpha_hat = np.minimum(rde, mde)

        self.var["_MoM_dispersions"] = np.full(self.n_vars, np.nan)
        self.var.loc[self.var["non_zero"], "_MoM_dispersions"] = np.clip(alpha_hat, self.min_disp, self.max_disp)

    def fit_genewise_with_normalization_factors(self, vst=False):
        if "normed_counts" not in self.layers:
            self.fit_size_factors(fit_type=self.size_factors_fit_type)

        self.var["non_zero"] = ~(self.X == 0).all(axis=0)
        self.non_zero_idx = np.arange(self.n_vars)[self.var["non_zero"]]
        self.non_zero_genes = self.var_names[self.var["non_zero"]]
        if isinstance(self.non_zero_genes, pd.MultiIndex):
            raise ValueError("non_zero_genes should not be a MultiIndex")

        self._fit_MoM_dispersions()
        design_matrix = self.obsm["design_matrix"].values
        norm_factors = self.obsm["normalization_factors"][:, self.non_zero_idx]
        counts = self.X[:, self.non_zero_idx]

        if len(self.obsm["design_matrix"].value_counts()) == self.obsm["design_matrix"].shape[-1]:
            mu_hat_ = self.inference.lin_reg_mu(
                counts=counts,
                size_factors=norm_factors,
                design_matrix=design_matrix,
                min_mu=self.min_mu,
            )
        else:
            _, mu_hat_, _, _ = self.inference.irls(
                counts=counts,
                size_factors=norm_factors,
                design_matrix=design_matrix,
                disp=self.var.loc[self.var["non_zero"], "_MoM_dispersions"].values,
                min_mu=self.min_mu,
                beta_tol=self.beta_tol,
            )

        mu_param_name = "_vst_mu_hat" if vst else "_mu_hat"
        disp_param_name = "vst_genewise_dispersions" if vst else "genewise_dispersions"
        self.layers[mu_param_name] = np.full((self.n_obs, self.n_vars), np.nan)
        self.layers[mu_param_name][:, self.var["non_zero"]] = mu_hat_

        dispersions_, converged_ = self.inference.alpha_mle(
            counts=counts,
            design_matrix=design_matrix,
            mu=self.layers[mu_param_name][:, self.non_zero_idx],
            alpha_hat=self.var.loc[self.var["non_zero"], "_MoM_dispersions"].values,
            min_disp=self.min_disp,
            max_disp=self.max_disp,
        )

        self.var[disp_param_name] = np.full(self.n_vars, np.nan)
        self.var.loc[self.var["non_zero"], disp_param_name] = np.clip(dispersions_, self.min_disp, self.max_disp)
        self.var["_genewise_converged"] = pd.array([pd.NA] * self.n_vars, dtype="boolean")
        self.var.loc[self.var["non_zero"], "_genewise_converged"] = converged_

    def fit_lfc_with_normalization_factors(self):
        if "dispersions" not in self.var:
            self.fit_MAP_dispersions()

        mle_lfcs_, mu_, hat_diagonals_, converged_ = self.inference.irls(
            counts=self.X[:, self.non_zero_idx],
            size_factors=self.obsm["normalization_factors"][:, self.non_zero_idx],
            design_matrix=self.obsm["design_matrix"].values,
            disp=self.var.loc[self.var["non_zero"], "dispersions"].values,
            min_mu=self.min_mu,
            beta_tol=self.beta_tol,
        )

        self.varm["LFC"] = pd.DataFrame(np.nan, index=self.var_names, columns=self.obsm["design_matrix"].columns)
        self.varm["LFC"].update(pd.DataFrame(mle_lfcs_, index=self.non_zero_genes, columns=self.obsm["design_matrix"].columns))
        self.obsm["_mu_LFC"] = mu_
        self.obsm["_hat_diagonals"] = hat_diagonals_
        self.var["_LFC_converged"] = pd.array([pd.NA] * self.n_vars, dtype="boolean")
        self.var.loc[self.var["non_zero"], "_LFC_converged"] = converged_

    def replace_outliers_with_normalization_factors(self):
        if "cooks" not in self.layers:
            self.calculate_cooks()

        num_samples = self.n_obs
        num_vars = self.obsm["design_matrix"].shape[1]
        self.obs["replaceable"] = utils.n_or_more_replicates(self.obsm["design_matrix"], self.min_replicates).values
        if self.obs["replaceable"].sum() == 0:
            self.var["replaced"] = np.full(self.n_vars, False)
            return

        cooks_cutoff = f.ppf(0.99, num_vars, num_samples - num_vars)
        idx = self.layers["cooks"] > cooks_cutoff
        self.var["replaced"] = idx.any(axis=0)
        if sum(self.var["replaced"] > 0):
            self.counts_to_refit = self[:, self.var["replaced"]].copy()
            norm_subset = self.obsm["normalization_factors"][:, self.var["replaced"]]
            trim_base_mean = pd.DataFrame(
                np.asarray(utils.trimmed_mean(self.counts_to_refit.X / norm_subset, trim=0.2, axis=0)),
                index=self.counts_to_refit.var_names,
            )
            replacement_counts = pd.DataFrame(
                trim_base_mean.values.T * norm_subset,
                index=self.counts_to_refit.obs_names,
                columns=self.counts_to_refit.var_names,
            ).astype(int)
            replace_mask = self.obs["replaceable"].values[:, None] & idx[:, self.var["replaced"]]
            self.counts_to_refit.X[replace_mask] = replacement_counts.values[replace_mask]

    def refit_without_outliers_with_normalization_factors(self):
        assert self.refit_cooks, "Trying to refit Cooks outliers but the 'refit_cooks' flag is set to False"
        if "replaced" not in self.var:
            self._replace_outliers()

        new_all_zeroes = (self.counts_to_refit.X == 0).all(axis=0)
        self.new_all_zeroes_genes = self.counts_to_refit.var_names[new_all_zeroes]
        self.var["refitted"] = self.var["replaced"].copy()
        self.var.loc[self.var["refitted"], "refitted"] = ~new_all_zeroes
        if new_all_zeroes.sum() > 0:
            self.var.loc[self.new_all_zeroes_genes, "_normed_means"] = 0
            self.varm["LFC"].loc[self.new_all_zeroes_genes, :] = 0
        if self.var["refitted"].sum() == 0:
            return

        self.counts_to_refit = self.counts_to_refit[:, ~new_all_zeroes].copy()
        if isinstance(self.new_all_zeroes_genes, pd.MultiIndex):
            raise ValueError

        sub_dds = DeseqDataSet(
            counts=pd.DataFrame(self.counts_to_refit.X, index=self.counts_to_refit.obs_names, columns=self.counts_to_refit.var_names),
            metadata=self.obs,
            design=self.design,
            min_mu=self.min_mu,
            min_disp=self.min_disp,
            max_disp=self.max_disp,
            refit_cooks=self.refit_cooks,
            min_replicates=self.min_replicates,
            beta_tol=self.beta_tol,
            inference=self.inference,
            quiet=self.quiet,
        )
        sub_norm_factors = pd.DataFrame(
            self.obsm["normalization_factors"][:, self.var["refitted"]],
            index=self.obs_names,
            columns=self.var_names[self.var["refitted"]],
        )
        install_transcript_length_normalization(sub_dds, sub_norm_factors, self.obs["size_factors"], opt)
        sub_dds.fit_genewise_dispersions()
        sub_dds.uns["disp_function_type"] = self.uns["disp_function_type"]
        if sub_dds.uns["disp_function_type"] == "parametric":
            sub_dds.uns["trend_coeffs"] = self.uns["trend_coeffs"]
        elif sub_dds.uns["disp_function_type"] == "mean":
            sub_dds.uns["mean_disp"] = self.uns["mean_disp"]
        sub_dds.var["_normed_means"] = sub_dds.layers["normed_counts"].mean(axis=0)
        sub_dds.var["fitted_dispersions"] = sub_dds.disp_function(sub_dds.var["_normed_means"])
        sub_dds.uns["_squared_logres"] = self.uns["_squared_logres"]
        sub_dds.uns["prior_disp_var"] = self.uns["prior_disp_var"]
        sub_dds.fit_MAP_dispersions()
        sub_dds.fit_LFC()

        self.var.loc[self.var["refitted"], "_normed_means"] = sub_dds.var["_normed_means"]
        self.varm["LFC"][self.var["refitted"]] = sub_dds.varm["LFC"]
        self.var.loc[self.var["refitted"], "genewise_dispersions"] = sub_dds.var["genewise_dispersions"]
        self.var.loc[self.var["refitted"], "fitted_dispersions"] = sub_dds.var["fitted_dispersions"]
        self.var.loc[self.var["refitted"], "dispersions"] = sub_dds.var["dispersions"]
        self.layers["replace_cooks"] = self.layers["cooks"].copy()
        for col in np.where(self.var["refitted"])[0]:
            self.layers["replace_cooks"][self.obs["replaceable"], col] = 0.0
        opt["transcript_length_cooks_refit_used_normalization_factors"] = True

    dds.fit_size_factors = types.MethodType(fit_size_factors_with_normalization_factors, dds)
    dds._fit_MoM_dispersions = types.MethodType(fit_moments_with_normalization_factors, dds)
    dds.fit_genewise_dispersions = types.MethodType(fit_genewise_with_normalization_factors, dds)
    dds.fit_LFC = types.MethodType(fit_lfc_with_normalization_factors, dds)
    dds._replace_outliers = types.MethodType(replace_outliers_with_normalization_factors, dds)
    dds._refit_without_outliers = types.MethodType(refit_without_outliers_with_normalization_factors, dds)


def install_r_style_parametric_dispersion_fit(dds, opt):
    if opt["fit_type"] != "parametric" or not opt["r_style_parametric_dispersion_fit"]:
        return

    original_parametric_fit = dds._fit_parametric_dispersion_trend

    def r_style_parametric_fit(self, vst=False):
        disp_param_name = "vst_genewise_dispersions" if vst else "genewise_dispersions"

        if disp_param_name not in self.var:
            self.fit_genewise_dispersions(vst)

        targets = pd.Series(
            self.var.loc[self.non_zero_genes, disp_param_name].copy(),
            index=self.non_zero_genes,
        )
        covariates = pd.Series(
            1 / self.var.loc[self.non_zero_genes, "_normed_means"],
            index=self.non_zero_genes,
        )

        min_disp = getattr(self, "min_disp", 1e-8)
        threshold = opt["parametric_dispersion_min_disp_multiplier"] * min_disp
        valid = pd.Series(
            np.isfinite(covariates.to_numpy())
            & np.isfinite(targets.to_numpy())
            & (targets.to_numpy() > threshold),
            index=targets.index,
        )

        if not vst:
            opt["parametric_dispersion_genes_before_r_style_filter"] = int(len(targets))
            opt["parametric_dispersion_genes_after_r_style_filter"] = int(valid.sum())

        targets = targets.loc[valid]
        covariates = covariates.loc[valid]

        if targets.empty:
            warnings.warn(
                "No genes remained after R-style parametric dispersion filtering. "
                "Switching to a mean-based dispersion trend.",
                UserWarning,
                stacklevel=2,
            )
            self._fit_mean_dispersion_trend(vst)
            return

        old_coeffs = pd.Series([0.1, 0.1])
        coeffs = pd.Series([1.0, 1.0])
        while (coeffs > 1e-10).all() and (
            np.log(np.abs(coeffs / old_coeffs)) ** 2
        ).sum() >= 1e-6:
            old_coeffs = coeffs
            coeffs, predictions, converged = self.inference.dispersion_trend_gamma_glm(
                covariates, targets
            )
            if not converged or (coeffs <= 1e-10).any():
                warnings.warn(
                    "The R-style dispersion trend curve fitting did not converge. "
                    "Switching to a mean-based dispersion trend.",
                    UserWarning,
                    stacklevel=2,
                )

                self._fit_mean_dispersion_trend(vst)
                return

            pred_ratios = self.var.loc[covariates.index, disp_param_name] / predictions
            keep = (pred_ratios >= 1e-4) & (pred_ratios < 15)
            targets = targets.loc[keep]
            covariates = covariates.loc[keep]

        if vst:
            self.uns["vst_trend_coeffs"] = pd.Series(coeffs, index=["a0", "a1"])
        else:
            self.uns["trend_coeffs"] = pd.Series(coeffs, index=["a0", "a1"])
            self.var["fitted_dispersions"] = np.full(self.n_vars, np.nan)
            self.uns["disp_function_type"] = "parametric"
            self.var.loc[self.var["non_zero"], "fitted_dispersions"] = (
                self.disp_function(self.var.loc[self.var["non_zero"], "_normed_means"])
            )

    dds._fit_parametric_dispersion_trend = types.MethodType(r_style_parametric_fit, dds)
    dds._pydeseq2_original_parametric_dispersion_trend = original_parametric_fit


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
    transcript_lengths = read_transcript_lengths(opt, count_table)
    normalization_factors, sample_size_factors = build_normalization_factors(count_table, transcript_lengths, opt)

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
    inference_class = NormalizationFactorInference if normalization_factors is not None else DefaultInference
    inference = inference_class(n_cpus=opt["cores"])
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
    install_transcript_length_normalization(dds, normalization_factors, sample_size_factors, opt)
    install_r_style_parametric_dispersion_fit(dds, opt)
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

    if normalization_factors is not None:
        norm_factor_output = matrix_to_output(
            normalization_factors.to_numpy(),
            normalization_factors.columns,
            normalization_factors.index,
            opt["gene_id_col"],
            opt["round_digits"],
        )
        write_tsv(norm_factor_output, f"{prefix}.pydeseq2.normalization_factors.tsv")

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
