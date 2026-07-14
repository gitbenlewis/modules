#!/usr/bin/env Rscript

parse_args <- function(args) {
    opts <- list(outdir = "work/pasilla_reference", prefilter = TRUE, cpus = 1L, repetitions = 1L, warmup = FALSE)
    i <- 1
    while (i <= length(args)) {
        key <- args[[i]]
        if (key == "--outdir") {
            i <- i + 1
            opts$outdir <- args[[i]]
        } else if (key == "--prefilter") {
            i <- i + 1
            opts$prefilter <- tolower(args[[i]]) %in% c("true", "t", "1", "yes")
        } else if (key == "--cpus") {
            i <- i + 1
            opts$cpus <- as.integer(strsplit(args[[i]], ",", fixed = TRUE)[[1]])
        } else if (key == "--repetitions") {
            i <- i + 1
            opts$repetitions <- as.integer(args[[i]])
        } else if (key == "--warmup") {
            i <- i + 1
            opts$warmup <- tolower(args[[i]]) %in% c("true", "t", "1", "yes")
        } else if (key %in% c("--help", "-h")) {
            cat(paste(
                "Usage: Rscript prepare_pasilla_reference.R [--outdir DIR]",
                "[--prefilter true|false] [--cpus 1,4] [--repetitions N] [--warmup true|false]\n"
            ))
            quit(status = 0)
        } else {
            stop("Unknown argument: ", key)
        }
        i <- i + 1
    }
    if (any(is.na(opts$cpus)) || any(opts$cpus < 1) || is.na(opts$repetitions) || opts$repetitions < 1) {
        stop("CPU values and repetitions must be positive integers")
    }
    opts
}

require_installed <- function(pkg) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        stop(
            "Missing R package '", pkg, "'. Install with: ",
            "if (!requireNamespace(\"BiocManager\", quietly = TRUE)) install.packages(\"BiocManager\"); ",
            "BiocManager::install(c(\"DESeq2\", \"BiocParallel\", \"pasilla\"))",
            call. = FALSE
        )
    }
}

write_tsv <- function(x, path) {
    utils::write.table(x, file = path, sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))
thread_limits <- c("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
if (any(Sys.getenv(thread_limits, unset = "") != "1")) {
    stop("Launch Rscript with OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, and VECLIB_MAXIMUM_THREADS=1")
}
require_installed("DESeq2")
require_installed("pasilla")
require_installed("BiocParallel")

script_start <- proc.time()
dir.create(opts$outdir, recursive = TRUE, showWarnings = FALSE)

pas_cts <- system.file("extdata", "pasilla_gene_counts.tsv.gz", package = "DESeq2", mustWork = TRUE)
pas_anno <- system.file("extdata", "pasilla_sample_annotation.csv", package = "DESeq2", mustWork = TRUE)

cts <- as.matrix(utils::read.csv(pas_cts, sep = "\t", row.names = "gene_id", check.names = FALSE))
coldata <- utils::read.csv(pas_anno, row.names = 1, check.names = FALSE)
coldata <- coldata[, c("condition", "type")]
rownames(coldata) <- sub("fb$", "", rownames(coldata))
coldata$condition <- stats::relevel(factor(coldata$condition), ref = "untreated")
coldata$type <- factor(coldata$type)

if (!all(rownames(coldata) %in% colnames(cts))) {
    stop("Sample annotation rows do not match count matrix columns")
}
cts <- cts[, rownames(coldata)]
rows_before_filter <- nrow(cts)

if (isTRUE(opts$prefilter)) {
    keep <- rowSums(cts >= 10) >= 3
    cts <- cts[keep, , drop = FALSE]
}

run_reference <- function(cpus) {
    dds <- DESeq2::DESeqDataSetFromMatrix(countData = cts, colData = coldata, design = ~ 0 + condition)
    dds$condition <- stats::relevel(dds$condition, ref = "untreated")
    deseq_runtime <- system.time({
        dds <- DESeq2::DESeq(
            dds,
            parallel = cpus > 1,
            BPPARAM = BiocParallel::MulticoreParam(cpus)
        )
        res <- DESeq2::results(dds, contrast = c("condition", "treated", "untreated"))
    })
    vst_runtime <- system.time({
        DESeq2::vst(dds, blind = TRUE)
    })
    list(dds = dds, results = res, deseq_seconds = deseq_runtime[["elapsed"]], vst_seconds = vst_runtime[["elapsed"]])
}

runtime_rows <- list()
last_run <- NULL
timer_runtime_version <- as.character(getRversion())
timer_engine_version <- as.character(utils::packageVersion("DESeq2"))
timer_host_system <- unname(Sys.info()[["sysname"]])
timer_host_machine <- unname(Sys.info()[["machine"]])
for (cpus in opts$cpus) {
    if (isTRUE(opts$warmup)) {
        run_reference(cpus)
    }
    for (repetition in seq_len(opts$repetitions)) {
        last_run <- run_reference(cpus)
        runtime_rows[[length(runtime_rows) + 1L]] <- data.frame(
            cpus = cpus,
            repetition = repetition,
            step = c("deseq_and_results", "vst"),
            elapsed_seconds = c(last_run$deseq_seconds, last_run$vst_seconds),
            timer_runtime = "R",
            timer_runtime_version = timer_runtime_version,
            timer_engine = "DESeq2",
            timer_engine_version = timer_engine_version,
            timer_host_system = timer_host_system,
            timer_host_machine = timer_host_machine,
            timer_dataset = "Pasilla",
            timer_genes = nrow(cts),
            timer_samples = ncol(cts),
            timer_design = "~ 0 + condition",
            timer_contrast = "condition_treated_vs_untreated",
            check.names = FALSE
        )
    }
}
res <- last_run$results

counts_out <- data.frame(gene_id = rownames(cts), cts, check.names = FALSE)
samples_out <- data.frame(
    sample = rownames(coldata),
    condition = as.character(coldata$condition),
    type = as.character(coldata$type),
    check.names = FALSE
)
contrasts_r_out <- data.frame(
    id = "condition_treated_vs_untreated",
    variable = "condition",
    reference = "untreated",
    target = "treated",
    formula = "~ 0 + condition",
    comparison = "conditiontreated - conditionuntreated",
    check.names = FALSE
)
contrasts_pydeseq2_out <- data.frame(
    id = "condition_treated_vs_untreated",
    variable = "condition",
    reference = "untreated",
    target = "treated",
    formula = "~ 0 + condition",
    comparison = "condition[treated] - condition[untreated]",
    check.names = FALSE
)
results_out <- data.frame(gene_id = rownames(res), as.data.frame(res), check.names = FALSE)
summary_out <- data.frame(
    prefilter = isTRUE(opts$prefilter),
    rows_before_filter = rows_before_filter,
    rows = nrow(results_out),
    na_pvalue = sum(is.na(results_out$pvalue)),
    na_padj = sum(is.na(results_out$padj)),
    design = "~ 0 + condition",
    contrast = "condition_treated_vs_untreated",
    native_math_threads_per_worker = 1L,
    check.names = FALSE
)
runtime_out <- do.call(rbind, runtime_rows)
summary_out$total_script_seconds <- unname((proc.time() - script_start)[["elapsed"]])

write_tsv(counts_out, file.path(opts$outdir, "pasilla_counts.tsv"))
write_tsv(samples_out, file.path(opts$outdir, "pasilla_samples.tsv"))
write_tsv(contrasts_r_out, file.path(opts$outdir, "pasilla_contrasts_r.tsv"))
write_tsv(contrasts_pydeseq2_out, file.path(opts$outdir, "pasilla_contrasts_pydeseq2.tsv"))
write_tsv(results_out, file.path(opts$outdir, "pasilla_deseq2_results.tsv"))
write_tsv(summary_out, file.path(opts$outdir, "pasilla_deseq2_summary.tsv"))
write_tsv(runtime_out, file.path(opts$outdir, "pasilla_deseq2_runtime.tsv"))

utils::capture.output(utils::sessionInfo(), file = file.path(opts$outdir, "pasilla_deseq2_sessionInfo.txt"))
