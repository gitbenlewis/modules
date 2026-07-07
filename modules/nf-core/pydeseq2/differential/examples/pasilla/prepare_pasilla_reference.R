#!/usr/bin/env Rscript

parse_args <- function(args) {
    opts <- list(outdir = "work/pasilla_reference", prefilter = TRUE)
    i <- 1
    while (i <= length(args)) {
        key <- args[[i]]
        if (key == "--outdir") {
            i <- i + 1
            opts$outdir <- args[[i]]
        } else if (key == "--prefilter") {
            i <- i + 1
            opts$prefilter <- tolower(args[[i]]) %in% c("true", "t", "1", "yes")
        } else if (key %in% c("--help", "-h")) {
            cat("Usage: Rscript prepare_pasilla_reference.R [--outdir DIR] [--prefilter true|false]\n")
            quit(status = 0)
        } else {
            stop("Unknown argument: ", key)
        }
        i <- i + 1
    }
    opts
}

require_installed <- function(pkg) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        stop(
            "Missing R package '", pkg, "'. Install with: ",
            "if (!requireNamespace(\"BiocManager\", quietly = TRUE)) install.packages(\"BiocManager\"); ",
            "BiocManager::install(c(\"DESeq2\", \"pasilla\"))",
            call. = FALSE
        )
    }
}

write_tsv <- function(x, path) {
    utils::write.table(x, file = path, sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))
require_installed("DESeq2")
require_installed("pasilla")

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

if (isTRUE(opts$prefilter)) {
    keep <- rowSums(cts >= 10) >= 3
    cts <- cts[keep, , drop = FALSE]
}

dds <- DESeq2::DESeqDataSetFromMatrix(countData = cts, colData = coldata, design = ~ condition)
dds$condition <- stats::relevel(dds$condition, ref = "untreated")
deseq_runtime <- system.time({
    dds <- DESeq2::DESeq(dds)
    res <- DESeq2::results(dds, name = "condition_treated_vs_untreated")
})

counts_out <- data.frame(gene_id = rownames(cts), cts, check.names = FALSE)
samples_out <- data.frame(
    sample = rownames(coldata),
    condition = as.character(coldata$condition),
    type = as.character(coldata$type),
    check.names = FALSE
)
contrasts_out <- data.frame(
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
    rows = nrow(results_out),
    na_pvalue = sum(is.na(results_out$pvalue)),
    na_padj = sum(is.na(results_out$padj)),
    design = "~ condition",
    contrast = "condition_treated_vs_untreated",
    check.names = FALSE
)
runtime_out <- data.frame(
    step = c("deseq_and_results", "total_script"),
    elapsed_seconds = c(unname(deseq_runtime[["elapsed"]]), unname((proc.time() - script_start)[["elapsed"]])),
    check.names = FALSE
)

write_tsv(counts_out, file.path(opts$outdir, "pasilla_counts.tsv"))
write_tsv(samples_out, file.path(opts$outdir, "pasilla_samples.tsv"))
write_tsv(contrasts_out, file.path(opts$outdir, "pasilla_contrasts.tsv"))
write_tsv(results_out, file.path(opts$outdir, "pasilla_deseq2_results.tsv"))
write_tsv(summary_out, file.path(opts$outdir, "pasilla_deseq2_summary.tsv"))
write_tsv(runtime_out, file.path(opts$outdir, "pasilla_deseq2_runtime.tsv"))

utils::capture.output(utils::sessionInfo(), file = file.path(opts$outdir, "pasilla_deseq2_sessionInfo.txt"))
