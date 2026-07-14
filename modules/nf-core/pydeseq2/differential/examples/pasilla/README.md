# Pasilla nf-core DESeq2 Comparison Example

This optional example runs the nf-core `deseq2/differential` and
`pydeseq2/differential` modules on the public Bioconductor pasilla count-matrix
dataset, then compares their outputs. It is intended for manual review and is
not part of nf-core CI.

Sources:

- Bioconductor pasilla package: https://bioconductor.org/packages/release/data/experiment/html/pasilla.html
- DESeq2 vignette: https://bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html

## 1. Create the comparison environment

The shared environment pins the R and Python engines, dataset packages,
comparison libraries, and Nextflow version used by this example:

```bash
conda env create --file ../environment.yml
conda activate nfcore-pydeseq2-comparison
```

The environment is only for these optional examples. It does not add runtime
dependencies to either nf-core module. Using one environment for the native
profile also gives both implementations the same OpenBLAS build.

## 2. Export pasilla inputs and an R core reference

From `modules/nf-core/pydeseq2/differential/examples/pasilla`:

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
Rscript prepare_pasilla_reference.R \
    --outdir work/pasilla_reference \
    --cpus 1,4 \
    --repetitions 7 \
    --warmup true
```

The launch-time limits keep each BiocParallel worker to one native math thread;
the preparation script fails if they are absent so a 4-worker run cannot
silently oversubscribe the host.

This writes:

- `pasilla_counts.tsv`: genes as rows and samples as columns.
- `pasilla_samples.tsv`: sample metadata with `sample`, `condition`, and `type`.
- `pasilla_contrasts_r.tsv`: treated-over-untreated contrast syntax for `deseq2/differential`.
- `pasilla_contrasts_pydeseq2.tsv`: treated-over-untreated contrast syntax for `pydeseq2/differential`.
- `pasilla_deseq2_results.tsv`: unshrunken R `DESeq2` Wald-test results.
- `pasilla_deseq2_runtime.tsv`: repeated core and VST timings at each CPU setting.
- `pasilla_deseq2_sessionInfo.txt`: R session provenance.

By default the script applies the vignette-style bulk RNA-seq prefilter
`rowSums(counts(dds) >= 10) >= 3` before running `DESeq()`. Use
`--prefilter false` to omit this prefilter and rely on downstream independent
filtering.

## 3. Run a paired smoke comparison

From `modules/nf-core/pydeseq2/differential/examples/pasilla`:

```bash
nextflow -C nextflow.config run run_pydeseq2_pasilla.nf \
    --input_dir work/pasilla_reference \
    --outdir work/nfcore_pasilla \
    --cpus 1 \
    --implementation both \
    -profile docker \
    -with-trace work/nfcore_pasilla/trace.tsv
```

Both module runs use the same count matrix, sample sheet, no-intercept model,
CPU allocation, and VST setting. The R module receives the limma-style comparison
`conditiontreated - conditionuntreated`, while the PyDESeq2 module receives the
formulaic comparison `condition[treated] - condition[untreated]`. The PyDESeq2
module also writes `*.pydeseq2.runtime.tsv`, which records major fitting,
statistics, VST, and aggregate script timings. In `both` mode, the PyDESeq2 task is
gated on the R results so the tasks cannot contend for resources. Use
`--implementation r` or `--implementation pydeseq2` to run one module.

The Docker profile uses each module's pinned container, keeps each module worker
to one native math thread, and applies a hard Docker CPU quota. On Apple silicon,
the pinned x86-64 images run under emulation and should not be interpreted as native performance.
Use the local profile with explicit environment paths for native measurements,
and retain the generated version and session records with the results.

## 4. Run the benchmark matrix

From this directory, run one warm-up and seven measured repetitions for each
module at 1 and 4 CPUs:

```bash
python ../run_benchmark_repetitions.py \
    --dataset Pasilla \
    --workflow run_pydeseq2_pasilla.nf \
    --config nextflow.config \
    --input-dir work/pasilla_reference \
    --outdir work/pasilla_benchmark \
    --cpus 1,4 \
    --repetitions 7 \
    --profile docker \
    --environment-yaml ../environment.yml
```

The Docker matrix intentionally omits the standalone R core timer: that timer
was measured in the native shared environment, whereas the module tasks use
their pinned containers. Mixing those environments would make the core-runtime
ratio invalid. The paired task and workflow timings remain comparable.

For native local environments, use explicit environment paths:

```bash
python ../run_benchmark_repetitions.py \
    --dataset Pasilla \
    --workflow run_pydeseq2_pasilla.nf \
    --config nextflow.config \
    --input-dir work/pasilla_reference \
    --outdir work/pasilla_benchmark \
    --cpus 1,4 \
    --repetitions 7 \
    --profile local \
    --r-env-bin "$CONDA_PREFIX/bin" \
    --pydeseq2-env-bin "$CONDA_PREFIX/bin" \
    --environment-yaml ../environment.yml \
    --r-core-runtime work/pasilla_reference/pasilla_deseq2_runtime.tsv
```

External R core timing is accepted only for this matched local setup: both
module bin paths must resolve to the same environment, and the timer's R,
DESeq2, operating-system, and architecture metadata must match the R task.

The runner counterbalances implementation order across repetitions and writes
`pasilla_benchmark_timings.tsv` with every measured task, task and workflow
timing sources, scoped phase timing, peak RSS where available, output size,
checksums, worker-count evidence, package versions, environment hash, Git state,
and host provenance. Each run has a unique work directory and does not use
Nextflow cache reuse. Native macOS runs leave peak RSS empty because Nextflow
does not report local task memory metrics without a container engine.

## 5. Compare R and PyDESeq2 outputs

The following compares the final 4-CPU repetitions from the matched local
environment and combines them with all seven measured timing rows:

```bash
python ../compare_deseq2_outputs.py \
    --dataset-name Pasilla \
    --output-prefix pasilla_4cpu \
    --cpus 4 \
    --r-results work/pasilla_benchmark/r_4cpu/run_7/results/r_deseq2/condition_treated_vs_untreated.deseq2.results.tsv \
    --pydeseq2-results work/pasilla_benchmark/pydeseq2_4cpu/run_7/results/pydeseq2/condition_treated_vs_untreated.pydeseq2.results.tsv \
    --r-runtime work/pasilla_reference/pasilla_deseq2_runtime.tsv \
    --pydeseq2-runtime work/pasilla_benchmark/pydeseq2_4cpu/run_7/results/pydeseq2/condition_treated_vs_untreated.pydeseq2.runtime.tsv \
    --pydeseq2-session-info work/pasilla_benchmark/pydeseq2_4cpu/run_7/results/pydeseq2/condition_treated_vs_untreated.Python_sessionInfo.log \
    --benchmark-timings work/pasilla_benchmark/pasilla_benchmark_timings.tsv \
    --r-size-factors work/pasilla_benchmark/r_4cpu/run_7/results/r_deseq2/condition_treated_vs_untreated.deseq2.sizefactors.tsv \
    --pydeseq2-size-factors work/pasilla_benchmark/pydeseq2_4cpu/run_7/results/pydeseq2/condition_treated_vs_untreated.pydeseq2.sizefactors.tsv \
    --r-normalised-counts work/pasilla_benchmark/r_4cpu/run_7/results/r_deseq2/condition_treated_vs_untreated.normalised_counts.tsv \
    --pydeseq2-normalised-counts work/pasilla_benchmark/pydeseq2_4cpu/run_7/results/pydeseq2/condition_treated_vs_untreated.normalised_counts.tsv \
    --r-vst-counts work/pasilla_benchmark/r_4cpu/run_7/results/r_deseq2/condition_treated_vs_untreated.vst.tsv \
    --pydeseq2-vst-counts work/pasilla_benchmark/pydeseq2_4cpu/run_7/results/pydeseq2/condition_treated_vs_untreated.vst.tsv \
    --outdir work/pasilla_compare_4cpu
```

The benchmark configs retain unrounded numerical output. The comparison reports
row counts, shared genes, numerical-difference quantiles, correlations, NA-mask
and sign disagreements, significant-gene overlap at `padj < 0.1`, and a
tie-aware top-N comparison. Optional paired matrix inputs add size-factor,
normalised-count, VST, per-sample, and PCA-geometry concordance. It also reports
two runtime comparisons:

- Core runtime: standalone R `DESeq()` plus `results()` versus PyDESeq2
  module-recorded `dds.deseq2()` plus `DeseqStats.summary()`.
- nf-core task runtime: `deseq2/differential` versus `pydeseq2/differential`
  process walltime from matched, counterbalanced benchmark runs and traces.

An explicitly supplied `--r-runtime` file is authoritative for the R core
median, while task medians and the PyDESeq2 core median come from the selected
benchmark table unless numeric overrides are supplied. The summary records the
source used for each runtime. `--r-runtime` requires both `--benchmark-timings`
and an explicit `--cpus`; its dataset, dimensions, design, contrast, runtime,
package, and host provenance must match the selected local benchmark.

For a Docker benchmark, omit `--r-runtime`; the report will leave the R core
comparison unavailable instead of combining native and container timings.

Scatter plots include summary-stat insets for quick visual review.

Treat these metrics as concordance checks, not proof of exact numerical
equivalence. DESeq2 documents several reasons for `NA` p-values or adjusted
p-values, including all-zero rows, Cook's distance outliers, and independent
filtering.

`compare_pasilla_outputs.py` remains as a compatibility entry point and forwards
the original Pasilla arguments to the shared dataset-neutral comparison tool.
