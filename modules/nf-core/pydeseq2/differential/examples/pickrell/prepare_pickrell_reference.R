#!/usr/bin/env Rscript

parse_args <- function(args) {
    opts <- list(outdir = "work/pickrell_reference", cpus = c(1L, 4L), repetitions = 3L, warmup = TRUE)
    i <- 1
    while (i <= length(args)) {
        key <- args[[i]]
        if (key == "--outdir") {
            i <- i + 1
            opts$outdir <- args[[i]]
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
                "Usage: Rscript prepare_pickrell_reference.R [--outdir DIR]",
                "[--cpus 1,4] [--repetitions N] [--warmup true|false]\n"
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
            "BiocManager::install(c(\"DESeq2\", \"BiocParallel\", \"Biobase\", \"tweeDEseqCountData\"))",
            call. = FALSE
        )
    }
}

write_tsv <- function(x, path) {
    utils::write.table(x, file = path, sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")
}

normalise_gender <- function(values) {
    gender <- tolower(trimws(as.character(values)))
    gender[gender %in% c("m", "man", "male")] <- "male"
    gender[gender %in% c("f", "woman", "female")] <- "female"
    if (any(!gender %in% c("female", "male"))) {
        stop("Pickrell gender metadata contains unsupported labels: ", paste(sort(unique(gender)), collapse = ", "))
    }
    factor(gender, levels = c("female", "male"))
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))
thread_limits <- c("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
if (any(Sys.getenv(thread_limits, unset = "") != "1")) {
    stop("Launch Rscript with OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, and VECLIB_MAXIMUM_THREADS=1")
}
for (pkg in c("DESeq2", "BiocParallel", "Biobase", "tweeDEseqCountData")) {
    require_installed(pkg)
}

script_start <- proc.time()
dir.create(opts$outdir, recursive = TRUE, showWarnings = FALSE)

data("pickrell", package = "tweeDEseqCountData", envir = environment())
if (!exists("pickrell.eset", inherits = FALSE)) {
    stop("The tweeDEseqCountData package did not provide pickrell.eset")
}

counts_all <- as.matrix(Biobase::exprs(pickrell.eset))
phenotype <- Biobase::pData(pickrell.eset)
gender_column <- intersect(c("gender", "Gender", "sex", "Sex"), colnames(phenotype))
if (length(gender_column) != 1) {
    stop("Expected exactly one gender or sex column in Pickrell phenotype metadata")
}
if (!identical(colnames(counts_all), rownames(phenotype))) {
    stop("Pickrell count columns and phenotype rows are not identically ordered")
}
if (any(!is.finite(counts_all)) || any(counts_all < 0) || any(counts_all != round(counts_all))) {
    stop("pickrell.eset must contain finite non-negative integer raw counts")
}

gender <- normalise_gender(phenotype[[gender_column]])
keep <- rowSums(counts_all) > 0
counts <- counts_all[keep, , drop = FALSE]
coldata <- data.frame(gender = gender, row.names = colnames(counts), check.names = FALSE)

run_reference <- function(cpus) {
    dds <- DESeq2::DESeqDataSetFromMatrix(countData = counts, colData = coldata, design = ~ 0 + gender)
    deseq_runtime <- system.time({
        dds <- DESeq2::DESeq(
            dds,
            parallel = cpus > 1,
            BPPARAM = BiocParallel::MulticoreParam(cpus)
        )
        res <- DESeq2::results(dds, contrast = c("gender", "male", "female"))
    })
    vst_runtime <- system.time({
        DESeq2::vst(dds, blind = TRUE)
    })
    list(results = res, deseq_seconds = deseq_runtime[["elapsed"]], vst_seconds = vst_runtime[["elapsed"]])
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
            timer_dataset = "Pickrell",
            timer_genes = nrow(counts),
            timer_samples = ncol(counts),
            timer_design = "~ 0 + gender",
            timer_contrast = "gender_male_vs_female",
            check.names = FALSE
        )
    }
}

counts_out <- data.frame(gene_id = rownames(counts), counts, check.names = FALSE)
samples_out <- data.frame(
    sample = rownames(coldata),
    gender = as.character(coldata$gender),
    check.names = FALSE
)
contrasts_r_out <- data.frame(
    id = "gender_male_vs_female",
    variable = "gender",
    reference = "female",
    target = "male",
    formula = "~ 0 + gender",
    comparison = "gendermale - genderfemale",
    check.names = FALSE
)
contrasts_pydeseq2_out <- data.frame(
    id = "gender_male_vs_female",
    variable = "gender",
    reference = "female",
    target = "male",
    formula = "~ 0 + gender",
    comparison = "gender[male] - gender[female]",
    check.names = FALSE
)
results_out <- data.frame(gene_id = rownames(last_run$results), as.data.frame(last_run$results), check.names = FALSE)
summary_out <- data.frame(
    source_object = "tweeDEseqCountData::pickrell.eset",
    normalisation = "none_raw_counts",
    rows_before_filter = nrow(counts_all),
    rows_after_all_zero_filter = nrow(counts),
    samples = ncol(counts),
    female_samples = sum(coldata$gender == "female"),
    male_samples = sum(coldata$gender == "male"),
    design = "~ 0 + gender",
    contrast = "gender_male_vs_female",
    native_math_threads_per_worker = 1L,
    total_script_seconds = unname((proc.time() - script_start)[["elapsed"]]),
    check.names = FALSE
)

write_tsv(counts_out, file.path(opts$outdir, "pickrell_counts.tsv"))
write_tsv(samples_out, file.path(opts$outdir, "pickrell_samples.tsv"))
write_tsv(contrasts_r_out, file.path(opts$outdir, "pickrell_contrasts_r.tsv"))
write_tsv(contrasts_pydeseq2_out, file.path(opts$outdir, "pickrell_contrasts_pydeseq2.tsv"))
write_tsv(results_out, file.path(opts$outdir, "pickrell_deseq2_results.tsv"))
write_tsv(summary_out, file.path(opts$outdir, "pickrell_deseq2_summary.tsv"))
write_tsv(do.call(rbind, runtime_rows), file.path(opts$outdir, "pickrell_deseq2_runtime.tsv"))
utils::capture.output(utils::sessionInfo(), file = file.path(opts$outdir, "pickrell_deseq2_sessionInfo.txt"))
