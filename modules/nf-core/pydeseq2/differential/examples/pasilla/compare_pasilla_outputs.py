#!/usr/bin/env python3

import runpy
import sys
from pathlib import Path


shared_script = Path(__file__).resolve().parents[1] / "compare_deseq2_outputs.py"
arguments = sys.argv[1:]
if not any(argument == "--outdir" or argument.startswith("--outdir=") for argument in arguments):
    arguments.extend(["--outdir", "work/pasilla_compare"])
sys.argv = [
    str(shared_script),
    "--dataset-name",
    "Pasilla",
    "--output-prefix",
    "pasilla",
    *arguments,
]
runpy.run_path(str(shared_script), run_name="__main__")
