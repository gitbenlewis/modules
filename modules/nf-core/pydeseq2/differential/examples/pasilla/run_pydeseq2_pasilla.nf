#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

include { PYDESEQ2_DIFFERENTIAL } from '../../main.nf'

params.input_dir = "work/pasilla_reference"
params.outdir = "work/pydeseq2_pasilla"
params.counts = "${params.input_dir}/pasilla_counts.tsv"
params.samples = "${params.input_dir}/pasilla_samples.tsv"
params.contrasts = "${params.input_dir}/pasilla_contrasts.tsv"

workflow {
    ch_contrasts = Channel
        .fromPath(params.contrasts, checkIfExists: true)
        .splitCsv(header: true, sep: '\t')
        .map { row -> tuple(row, row.variable, row.reference, row.target, row.formula, row.comparison) }

    ch_inputs = Channel.value([
        [id: 'pasilla'],
        file(params.samples, checkIfExists: true),
        file(params.counts, checkIfExists: true),
    ])

    PYDESEQ2_DIFFERENTIAL(
        ch_contrasts,
        ch_inputs,
        Channel.value([[:], []]),
        Channel.value([[:], []])
    )
}
