process DESEQ2_PYDGE {
    tag "$meta.id"
    label 'process_single'

    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine in ['singularity', 'apptainer'] && !task.ext.singularity_pull_docker_container ?
        'https://depot.galaxyproject.org/singularity/pydeseq2:0.5.4--pyhdfd78af_0' :
        'quay.io/biocontainers/pydeseq2:0.5.4--pyhdfd78af_0' }"

    input:
    tuple val(meta), val(contrast_variable), val(reference), val(target), val(formula), val(comparison)
    tuple val(meta2), path(samplesheet), path(counts)
    tuple val(meta3), path(control_genes_file)
    tuple val(meta4), path(transcript_lengths_file)

    output:
    tuple val(meta), path("*.pydeseq2.results.tsv")        , emit: results
    tuple val(meta), path("*.pydeseq2.dispersion.png")     , emit: dispersion_plot_png
    tuple val(meta), path("*.pydeseq2.dispersion.pdf")     , emit: dispersion_plot_pdf
    tuple val(meta), path("*.pydeseq2.dds.h5ad")           , emit: h5ad
    tuple val(meta), path("*.pydeseq2.sizefactors.tsv")    , emit: size_factors
    tuple val(meta), path("*.normalised_counts.tsv")       , emit: normalised_counts
    tuple val(meta), path("*.vst.tsv")                     , optional: true, emit: vst_counts
    tuple val(meta), path("*.pydeseq2.model.txt")          , emit: model
    tuple val(meta), path("*.Python_sessionInfo.log")      , emit: session_info
    path "versions.yml"                                    , emit: versions, topic: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    prefix = task.ext.prefix ?: "${meta.id}"
    args = task.ext.args ?: ''
    template 'deseq2_pydge.py'

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.pydeseq2.results.tsv
    touch ${prefix}.pydeseq2.dispersion.png
    touch ${prefix}.pydeseq2.dispersion.pdf
    touch ${prefix}.pydeseq2.dds.h5ad
    touch ${prefix}.pydeseq2.sizefactors.tsv
    touch ${prefix}.normalised_counts.tsv
    touch ${prefix}.vst.tsv
    touch ${prefix}.pydeseq2.model.txt
    touch ${prefix}.Python_sessionInfo.log

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python -c 'import platform; print(platform.python_version())')
        pydeseq2: \$(python -c 'from importlib.metadata import version; print(version("pydeseq2"))')
    END_VERSIONS
    """
}
