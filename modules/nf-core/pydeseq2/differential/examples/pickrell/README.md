# Pickrell nf-core DESeq2 Scaling Benchmark

This optional benchmark compares `deseq2/differential` and
`pydeseq2/differential` on the raw Pickrell RNA-seq count matrix distributed in
Bioconductor's MIT-licensed `tweeDEseqCountData` package. It complements the
smaller Pasilla example with approximately 52,580 genes and 69 samples.
Generated data and benchmark outputs are not part of nf-core CI and must not be
committed.

Sources:

- Bioconductor data package: https://bioconductor.org/packages/tweeDEseqCountData/
- Pickrell et al. Nature 2010: https://doi.org/10.1038/nature08872

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

## 2. Prepare raw inputs and R core references

From `modules/nf-core/pydeseq2/differential/examples/pickrell`:

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
Rscript prepare_pickrell_reference.R \
    --outdir work/pickrell_reference \
    --cpus 1,4 \
    --repetitions 7 \
    --warmup true
```

The launch-time limits keep each BiocParallel worker to one native math thread;
the preparation script fails if they are absent so a 4-worker run cannot
silently oversubscribe the host.

The script exports only `pickrell.eset` raw counts, normalises the source sex
labels to `female` and `male`, removes genes whose counts are zero in all 69
samples, and records the original and retained dimensions. It writes:

- `pickrell_counts.tsv`: raw integer counts after the all-zero filter.
- `pickrell_samples.tsv`: stable `sample` and `gender` metadata.
- `pickrell_contrasts_r.tsv`: R module contrast syntax.
- `pickrell_contrasts_pydeseq2.tsv`: PyDESeq2 formulaic contrast syntax.
- `pickrell_deseq2_results.tsv`: unshrunken R reference results.
- `pickrell_deseq2_runtime.tsv`: repeated R core and VST timings.
- `pickrell_deseq2_summary.tsv`: dimensions, group sizes, design, and provenance.
- `pickrell_deseq2_sessionInfo.txt`: R package and runtime provenance.

Both implementations receive `~ 0 + gender` and test male versus female.

## 3. Run a paired smoke comparison

```bash
nextflow -C nextflow.config run run_pickrell_comparison.nf \
    --input_dir work/pickrell_reference \
    --outdir work/nfcore_pickrell \
    --cpus 1 \
    --implementation both \
    -profile docker \
    -with-trace work/nfcore_pickrell/trace.tsv
```

The PyDESeq2 task is gated on the R result in `both` mode, so the tasks do not
compete for CPU or memory. Both tasks use VST, unshrunken LFCs, the same retained
genes, the same CPU allocation, and their engine-specific equivalent contrasts.
The Docker profile keeps each module worker to one native math thread and
applies a hard CPU quota. On Apple
silicon, the pinned x86-64 images run under emulation; use the local profile for
native performance measurements and retain its generated provenance records.

## 4. Run the 1- and 4-CPU benchmark matrix

```bash
python ../run_benchmark_repetitions.py \
    --dataset Pickrell \
    --workflow run_pickrell_comparison.nf \
    --config nextflow.config \
    --input-dir work/pickrell_reference \
    --outdir work/pickrell_benchmark \
    --cpus 1,4 \
    --repetitions 7 \
    --profile docker \
    --environment-yaml ../environment.yml
```

The Docker matrix intentionally omits the standalone R core timer because it
was measured in the native shared environment, while the module tasks use their
pinned containers. The paired task and workflow timings remain comparable.

For native local environments, use explicit environment paths:

```bash
python ../run_benchmark_repetitions.py \
    --dataset Pickrell \
    --workflow run_pickrell_comparison.nf \
    --config nextflow.config \
    --input-dir work/pickrell_reference \
    --outdir work/pickrell_benchmark \
    --cpus 1,4 \
    --repetitions 7 \
    --profile local \
    --r-env-bin "$CONDA_PREFIX/bin" \
    --pydeseq2-env-bin "$CONDA_PREFIX/bin" \
    --environment-yaml ../environment.yml \
    --r-core-runtime work/pickrell_reference/pickrell_deseq2_runtime.tsv
```

External R core timing is accepted only for this matched local setup: both
module bin paths must resolve to the same environment, and the timer's R,
DESeq2, operating-system, and architecture metadata must match the R task.

The runner counterbalances implementation order across repetitions and writes
`pickrell_benchmark_timings.tsv` with every measured task, task and workflow
timing sources, scoped phase timing, peak RSS where available, output size,
checksums, worker-count evidence, package versions, environment hash, Git state,
and host provenance. Each run has a unique work directory and does not use
Nextflow cache reuse. Native macOS runs leave peak RSS empty because Nextflow
does not report local task memory metrics without a container engine.

## 5. Generate the concordance report

For the matched local environment:

```bash
python ../compare_deseq2_outputs.py \
    --dataset-name Pickrell \
    --output-prefix pickrell_4cpu \
    --cpus 4 \
    --r-results work/pickrell_benchmark/r_4cpu/run_7/results/r_deseq2/gender_male_vs_female.deseq2.results.tsv \
    --pydeseq2-results work/pickrell_benchmark/pydeseq2_4cpu/run_7/results/pydeseq2/gender_male_vs_female.pydeseq2.results.tsv \
    --r-runtime work/pickrell_reference/pickrell_deseq2_runtime.tsv \
    --pydeseq2-runtime work/pickrell_benchmark/pydeseq2_4cpu/run_7/results/pydeseq2/gender_male_vs_female.pydeseq2.runtime.tsv \
    --pydeseq2-session-info work/pickrell_benchmark/pydeseq2_4cpu/run_7/results/pydeseq2/gender_male_vs_female.Python_sessionInfo.log \
    --benchmark-timings work/pickrell_benchmark/pickrell_benchmark_timings.tsv \
    --r-size-factors work/pickrell_benchmark/r_4cpu/run_7/results/r_deseq2/gender_male_vs_female.deseq2.sizefactors.tsv \
    --pydeseq2-size-factors work/pickrell_benchmark/pydeseq2_4cpu/run_7/results/pydeseq2/gender_male_vs_female.pydeseq2.sizefactors.tsv \
    --r-normalised-counts work/pickrell_benchmark/r_4cpu/run_7/results/r_deseq2/gender_male_vs_female.normalised_counts.tsv \
    --pydeseq2-normalised-counts work/pickrell_benchmark/pydeseq2_4cpu/run_7/results/pydeseq2/gender_male_vs_female.normalised_counts.tsv \
    --r-vst-counts work/pickrell_benchmark/r_4cpu/run_7/results/r_deseq2/gender_male_vs_female.vst.tsv \
    --pydeseq2-vst-counts work/pickrell_benchmark/pydeseq2_4cpu/run_7/results/pydeseq2/gender_male_vs_female.vst.tsv \
    --outdir work/pickrell_compare_4cpu
```

Repeat with `--cpus 1` and the corresponding `1cpu` result paths for the serial
comparison. The benchmark configs retain unrounded numerical output. The report
includes numerical-difference quantiles, correlations, NA-mask and sign
disagreements, significant-gene overlap at `padj < 0.1`, a tie-aware top-N
comparison, per-run timings, and robust timing summaries. Optional paired
matrix inputs add size-factor, normalised-count, VST, per-sample, and
PCA-geometry concordance. Treat these as concordance and scaling evidence, not
proof of exact numerical equivalence or a requirement that either engine be
faster.

`--r-runtime` requires both `--benchmark-timings` and an explicit `--cpus`; its
dataset, dimensions, design, contrast, runtime, package, and host provenance
must match the selected local benchmark. For a Docker benchmark, omit
`--r-runtime`; the report will leave the R core
comparison unavailable instead of combining native and container timings.

## 6. Summarise Pasilla and Pickrell scaling

From the repository root, combine the two per-run tables:

```bash
python modules/nf-core/pydeseq2/differential/examples/summarise_scaling.py \
    --timings path/to/pasilla_benchmark_timings.tsv path/to/pickrell_benchmark_timings.tsv \
    --r-core-runtime Pasilla=path/to/pasilla_deseq2_runtime.tsv \
    --r-core-runtime Pickrell=path/to/pickrell_deseq2_runtime.tsv \
    --outdir work/pydeseq2_scaling_summary
```

The summary validates the complete dataset-by-implementation-by-CPU matrix and
contains per-setting medians, interquartile ranges, median absolute deviations,
1-to-4 CPU speedups, parallel efficiency, deterministic-output checks, package,
environment, Git, and host provenance, scoped PyDESeq2 phase summaries, and an
explicit statement of whether the faster implementation changes across tested
datasets or CPU settings. Dataset differences are not attributed to size alone.
The optional R core-runtime arguments are for matched local environments when
standalone references are timed in a separate uncontended pass. The summary
validates timer-specific version and host provenance, then writes the effective
values and relative provenance paths to `benchmark_effective_timings.tsv`.
