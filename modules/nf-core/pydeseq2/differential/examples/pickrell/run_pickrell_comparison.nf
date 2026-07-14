#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

include { DESEQ2_DIFFERENTIAL } from '../../../../deseq2/differential/main.nf'
include { PYDESEQ2_DIFFERENTIAL } from '../../main.nf'

params.input_dir = "work/pickrell_reference"
params.outdir = "work/nfcore_pickrell"
params.counts = "${params.input_dir}/pickrell_counts.tsv"
params.samples = "${params.input_dir}/pickrell_samples.tsv"
params.r_contrasts = "${params.input_dir}/pickrell_contrasts_r.tsv"
params.pydeseq2_contrasts = "${params.input_dir}/pickrell_contrasts_pydeseq2.tsv"
params.cpus = 1
params.implementation = "both"

workflow {
    if (!(params.implementation in ['r', 'pydeseq2', 'both'])) {
        error "--implementation must be one of: r, pydeseq2, both"
    }

    r_input = [
        [id: 'pickrell'],
        file(params.samples, checkIfExists: true),
        file(params.counts, checkIfExists: true),
    ]
    pydeseq2_input = [
        [id: 'pickrell'],
        file(params.samples, checkIfExists: true),
        file(params.counts, checkIfExists: true),
    ]

    if (params.implementation in ['r', 'both']) {
        ch_r_contrasts = channel
            .fromPath(params.r_contrasts, checkIfExists: true)
            .splitCsv(header: true, sep: '\t')
            .map { row -> tuple(row, row.variable, row.reference, row.target, row.formula, row.comparison) }

        DESEQ2_DIFFERENTIAL(
            ch_r_contrasts,
            channel.value(r_input),
            channel.value([[:], []]),
            channel.value([[:], []])
        )
    }

    if (params.implementation in ['pydeseq2', 'both']) {
        ch_pydeseq2_contrasts = channel
            .fromPath(params.pydeseq2_contrasts, checkIfExists: true)
            .splitCsv(header: true, sep: '\t')
            .map { row -> tuple(row, row.variable, row.reference, row.target, row.formula, row.comparison) }

        ch_pydeseq2_inputs = params.implementation == 'both' \
            ? DESEQ2_DIFFERENTIAL.out.results.map { pydeseq2_input } \
            : channel.value(pydeseq2_input)

        PYDESEQ2_DIFFERENTIAL(
            ch_pydeseq2_contrasts,
            ch_pydeseq2_inputs,
            channel.value([[:], []]),
            channel.value([[:], []])
        )
    }
}
