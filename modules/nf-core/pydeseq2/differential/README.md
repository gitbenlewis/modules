# pydeseq2/differential

This module runs differential expression analysis with PyDESeq2 and emits
PyDESeq2-native outputs. It intentionally does not fake R-only DESeq2 outputs or
features such as RDS objects, rlog, or ASHR shrinkage.

## Pasilla comparison example

An optional R-vs-PyDESeq2 concordance example is available in
`examples/pasilla`. It uses the public Bioconductor pasilla dataset and the
standard DESeq2 treated-over-untreated comparison described in the DESeq2
vignette.

The example is not part of nf-core module CI because it requires R/Bioconductor
packages (`DESeq2`, `pasilla`, and `BiocManager`) and is intended as a
reviewer/user reproducibility aid rather than a lightweight module unit test.

See `examples/pasilla/README.md` for setup, run commands, and comparison
outputs.
