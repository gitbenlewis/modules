# Pasilla R DESeq2 Comparison Example

This optional example runs `pydeseq2/differential` on the public Bioconductor
pasilla count-matrix dataset and compares PyDESeq2 output with an R `DESeq2`
reference. It is intended for manual review and is not part of nf-core CI.

Sources:

- Bioconductor pasilla package: https://bioconductor.org/packages/release/data/experiment/html/pasilla.html
- DESeq2 vignette: https://bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html

## 1. Install R-side comparison dependencies

Run this in R if the packages are not already installed:

```r
if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager")
}
BiocManager::install(c("DESeq2", "pasilla"))
```

These packages are only needed for the optional R reference/example. They are
not module runtime dependencies.

## 2. Export pasilla inputs and R reference output

From `modules/nf-core/pydeseq2/differential/examples/pasilla`:

```bash
Rscript prepare_pasilla_reference.R --outdir work/pasilla_reference
```

This writes:

- `pasilla_counts.tsv`: genes as rows and samples as columns.
- `pasilla_samples.tsv`: sample metadata with `sample`, `condition`, and `type`.
- `pasilla_contrasts.tsv`: one treated-over-untreated contrast for the module.
- `pasilla_deseq2_results.tsv`: unshrunken R `DESeq2` Wald-test results.
- `pasilla_deseq2_runtime.tsv`: elapsed seconds for `DESeq()` plus `results()`.
- `pasilla_deseq2_sessionInfo.txt`: R session provenance.

By default the script applies the vignette-style bulk RNA-seq prefilter
`rowSums(counts(dds) >= 10) >= 3` before running `DESeq()`. Use
`--prefilter false` to omit this prefilter and rely on downstream independent
filtering.

## 3. Run the PyDESeq2 module

From `modules/nf-core/pydeseq2/differential/examples/pasilla`:

```bash
nextflow run run_pydeseq2_pasilla.nf \
    --input_dir work/pasilla_reference \
    --outdir work/pydeseq2_pasilla \
    -profile conda \
    -with-trace work/pydeseq2_pasilla/trace.tsv
```

The R reference uses `design = ~ condition` with `untreated` as the reference.
For PyDESeq2/formulaic compatibility, the example module run uses the equivalent
no-intercept model `~ 0 + condition` and the explicit contrast
`condition[treated] - condition[untreated]`.

## 4. Compare R and PyDESeq2 outputs

```bash
python compare_pasilla_outputs.py \
    --r-results work/pasilla_reference/pasilla_deseq2_results.tsv \
    --pydeseq2-results work/pydeseq2_pasilla/condition_treated_vs_untreated.pydeseq2.results.tsv \
    --r-runtime work/pasilla_reference/pasilla_deseq2_runtime.tsv \
    --pydeseq2-trace work/pydeseq2_pasilla/trace.tsv \
    --outdir work/pasilla_compare
```

The comparison reports row counts, shared genes, log2 fold-change correlation,
p-value and adjusted-p-value rank correlations, sign concordance, significant
gene overlap at `padj < 0.1`, top-N overlap, R/PyDESeq2 runtime seconds, and
the PyDESeq2-over-R runtime ratio. It also reports how many rows have `NA`
p-values or adjusted p-values in each implementation. Scatter plots include
summary-stat insets for quick visual review.

Treat these metrics as concordance checks, not proof of exact numerical
equivalence. DESeq2 documents several reasons for `NA` p-values or adjusted
p-values, including all-zero rows, Cook's distance outliers, and independent
filtering.
