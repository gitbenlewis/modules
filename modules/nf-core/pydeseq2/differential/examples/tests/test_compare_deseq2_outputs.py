import importlib.util
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "compare_deseq2_outputs.py"
SPEC = importlib.util.spec_from_file_location("compare_deseq2_outputs", SCRIPT)
COMPARATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARATOR)


class TieAwareRankingTests(unittest.TestCase):
    def test_rounded_zero_block_is_included_without_gene_id_tiebreak(self):
        frame = pd.DataFrame(
            {
                "gene_id": ["z_gene", "a_gene", "m_gene", "next_gene"],
                "padj": [0.0, 0.0, 0.0, 0.1],
                "pvalue": [0.0, 0.0, 0.0, 0.01],
            }
        )

        ranked = COMPARATOR.inclusive_top_ranked_genes(frame, 2)

        self.assertEqual(ranked["genes"], {"z_gene", "a_gene", "m_gene"})
        self.assertEqual(ranked["included"], 3)
        self.assertEqual(ranked["boundary_tie_size"], 3)
        self.assertEqual(ranked["boundary_tie_expansion"], 1)
        self.assertTrue(ranked["table"]["boundary_member"].all())
        self.assertEqual(set(ranked["table"]["rank_min"]), {1})
        self.assertEqual(set(ranked["table"]["rank_max"]), {3})

    def test_pvalue_breaks_padj_ties_but_its_boundary_ties_remain_inclusive(self):
        frame = pd.DataFrame(
            {
                "gene_id": ["first", "tie_b", "tie_a", "last"],
                "padj": [0.0, 0.0, 0.0, 0.0],
                "pvalue": [0.01, 0.02, 0.02, 0.03],
            }
        )

        ranked = COMPARATOR.inclusive_top_ranked_genes(frame, 2)

        self.assertEqual(ranked["genes"], {"first", "tie_a", "tie_b"})
        self.assertEqual(ranked["boundary_pvalue"], 0.02)
        self.assertEqual(ranked["boundary_tie_size"], 2)


class ResultDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.r = pd.DataFrame(
            {
                "gene_id": ["g1", "g2", "g3"],
                "baseMean": [10.0, 20.0, 30.0],
                "log2FoldChange": [1.0, -1.0, 2.0],
                "pvalue": [0.01, np.nan, 0.3],
                "padj": [0.09, 0.2, 0.4],
            }
        )
        self.py = pd.DataFrame(
            {
                "gene_id": ["g1", "g2", "g3"],
                "baseMean": [10.0, 21.0, 30.0],
                "log2FoldChange": [1.0, -1.0, -2.0],
                "pvalue": [0.01, 0.2, 0.3],
                "padj": [0.11, 0.2, 0.4],
            }
        )
        r_prefixed = self.r.add_prefix("r_").rename(columns={"r_gene_id": "gene_id"})
        py_prefixed = self.py.add_prefix("py_").rename(columns={"py_gene_id": "gene_id"})
        self.joined = r_prefixed.merge(py_prefixed, on="gene_id", validate="one_to_one")

    def test_na_sign_and_significance_disagreements_are_explicit(self):
        numeric, na_rows, _ = COMPARATOR.numeric_difference_diagnostics(self.r, self.py, self.joined)
        pvalue = numeric.set_index("column").loc["pvalue"]
        self.assertEqual(pvalue["na_mask_disagreements"], 1)
        self.assertEqual(na_rows.loc[0, "gene_id"], "g2")
        self.assertEqual(na_rows.loc[0, "column"], "pvalue")

        sign_rows = COMPARATOR.sign_disagreement_table(self.joined)
        self.assertEqual(sign_rows["gene_id"].tolist(), ["g3"])

        significance_rows = COMPARATOR.significance_disagreement_table(self.r, self.py, alpha=0.1)
        self.assertEqual(significance_rows["gene_id"].tolist(), ["g1"])
        self.assertTrue(significance_rows.iloc[0]["r_significant"])
        self.assertFalse(significance_rows.iloc[0]["pydeseq2_significant"])


class CommandLineContractTests(unittest.TestCase):
    def setUp(self):
        self.base_args = [
            "--dataset-name",
            "Pasilla",
            "--output-prefix",
            "pasilla",
            "--r-results",
            "r.tsv",
            "--pydeseq2-results",
            "py.tsv",
            "--outdir",
            "report",
        ]

    def test_concordance_only_mode_does_not_require_benchmark_timings(self):
        args = COMPARATOR.parse_args(self.base_args)

        self.assertIsNone(args.benchmark_timings)

    def test_r_runtime_requires_benchmark_timings(self):
        with self.assertRaises(SystemExit):
            COMPARATOR.parse_args(
                [
                    *self.base_args,
                    "--cpus",
                    "1",
                    "--r-runtime",
                    "r-runtime.tsv",
                ]
            )

    def test_r_runtime_requires_explicit_cpu_selection(self):
        with self.assertRaises(SystemExit):
            COMPARATOR.parse_args(
                [
                    *self.base_args,
                    "--benchmark-timings",
                    "benchmark.tsv",
                    "--r-runtime",
                    "r-runtime.tsv",
                ]
            )


class BenchmarkSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.path = Path(self.tempdir.name) / "benchmark.tsv"
        pd.DataFrame(
            {
                "dataset": ["Pasilla", "Pasilla", "Pasilla", "Pasilla"],
                "implementation": ["r", "pydeseq2", "r", "pydeseq2"],
                "cpus": [1, 1, 4, 4],
                "repetition": [1, 1, 1, 1],
                "wall_seconds": [7.0, 17.0, 10.0, 14.0],
                "genes": [8148, 8148, 8148, 8148],
                "samples": [7, 7, 7, 7],
                "design": ["~ 0 + condition"] * 4,
                "contrast": ["condition_treated_vs_untreated"] * 4,
                "profile": ["local"] * 4,
            }
        ).to_csv(self.path, sep="\t", index=False)

    def test_multiple_cpu_values_require_explicit_cpu_selection(self):
        with self.assertRaisesRegex(ValueError, "--cpus is required"):
            COMPARATOR.read_benchmark_timings(self.path, "Pasilla", None)

    def test_empty_dataset_and_cpu_selections_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "no rows for dataset 'Pickrell'"):
            COMPARATOR.read_benchmark_timings(self.path, "Pickrell", 1)
        with self.assertRaisesRegex(ValueError, "no rows for dataset 'Pasilla' at 8 CPUs"):
            COMPARATOR.read_benchmark_timings(self.path, "Pasilla", 8)

    def test_explicit_cpu_selects_one_complete_slice(self):
        selected = COMPARATOR.read_benchmark_timings(self.path, "pasilla", 4)

        self.assertEqual(set(selected["cpus"]), {4})
        self.assertEqual(set(selected["implementation"]), {"r", "pydeseq2"})

    def test_identity_fields_are_required_only_for_external_r_runtime(self):
        legacy = pd.read_csv(self.path, sep="\t")[
            ["dataset", "implementation", "cpus", "repetition", "wall_seconds"]
        ]
        legacy.to_csv(self.path, sep="\t", index=False)

        selected = COMPARATOR.read_benchmark_timings(self.path, "Pasilla", 1)

        self.assertEqual(set(selected["implementation"]), {"r", "pydeseq2"})


class RuntimePrecedenceTests(unittest.TestCase):
    @staticmethod
    def args(**overrides):
        values = {
            "r_runtime": None,
            "cpus": 1,
            "pydeseq2_runtime_seconds": None,
            "pydeseq2_runtime": None,
            "pydeseq2_session_info": None,
            "r_task_seconds": None,
            "pydeseq2_task_seconds": None,
            "nextflow_trace": None,
        }
        values.update(overrides)
        return Namespace(**values)

    @staticmethod
    def benchmark_timings():
        return pd.DataFrame(
            {
                "dataset": ["Pasilla", "Pasilla"],
                "implementation": ["r", "pydeseq2"],
                "core_seconds": [101.0, 202.0],
                "wall_seconds": [303.0, 404.0],
                "profile": ["local", "local"],
                "r_version": ["4.5.3", None],
                "deseq2_version": ["1.50.2", None],
                "host_system": ["Darwin", "Darwin"],
                "host_machine": ["arm64", "arm64"],
                "genes": [8148, 8148],
                "samples": [7, 7],
                "design": ["~ 0 + condition", "~ 0 + condition"],
                "contrast": ["condition_treated_vs_untreated", "condition_treated_vs_untreated"],
            }
        )

    def test_standalone_r_runtime_file_beats_benchmark_core_timing(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime_path = Path(tempdir) / "r_runtime.tsv"
            runtime_path.write_text(
                "cpus\trepetition\tstep\telapsed_seconds\ttimer_runtime\t"
                "timer_runtime_version\ttimer_engine\ttimer_engine_version\t"
                "timer_host_system\ttimer_host_machine\ttimer_dataset\ttimer_genes\t"
                "timer_samples\ttimer_design\ttimer_contrast\n"
                "1\t1\tdeseq_and_results\t7\tR\t4.5.3\tDESeq2\t1.50.2\tDarwin\tarm64\t"
                "Pasilla\t8148\t7\t~ 0 + condition\tcondition_treated_vs_untreated\n"
                "1\t2\tdeseq_and_results\t9\tR\t4.5.3\tDESeq2\t1.50.2\tDarwin\tarm64\t"
                "Pasilla\t8148\t7\t~ 0 + condition\tcondition_treated_vs_untreated\n",
                encoding="utf-8",
            )
            args = self.args(r_runtime=str(runtime_path))

            resolved = COMPARATOR.resolve_runtime_inputs(args, self.benchmark_timings())

        self.assertEqual(resolved["r_core_runtime_seconds"], 8.0)
        self.assertEqual(resolved["r_core_runtime_source"], "r_runtime_file")

    def test_external_r_runtime_rejects_container_benchmark(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime_path = Path(tempdir) / "r_runtime.tsv"
            runtime_path.write_text("cpus\trepetition\tstep\telapsed_seconds\n", encoding="utf-8")
            timings = self.benchmark_timings().assign(profile="docker")

            with self.assertRaisesRegex(ValueError, "matched local"):
                COMPARATOR.read_r_runtime(runtime_path, 1, timings)

    def test_external_r_runtime_requires_the_selected_cpu(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime_path = Path(tempdir) / "r_runtime.tsv"
            runtime_path.write_text(
                "cpus\trepetition\tstep\telapsed_seconds\ttimer_runtime\t"
                "timer_runtime_version\ttimer_engine\ttimer_engine_version\t"
                "timer_host_system\ttimer_host_machine\ttimer_dataset\ttimer_genes\t"
                "timer_samples\ttimer_design\ttimer_contrast\n"
                "4\t1\tdeseq_and_results\t7\tR\t4.5.3\tDESeq2\t1.50.2\tDarwin\tarm64\t"
                "Pasilla\t8148\t7\t~ 0 + condition\tcondition_treated_vs_untreated\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "no 1-CPU timing rows"):
                COMPARATOR.read_r_runtime(runtime_path, 1, self.benchmark_timings())

    def test_external_r_runtime_requires_explicit_cpu_selection(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime_path = Path(tempdir) / "r_runtime.tsv"
            runtime_path.write_text("placeholder\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "--cpus is required with --r-runtime"):
                COMPARATOR.read_r_runtime(runtime_path, None, self.benchmark_timings())

    def test_external_r_runtime_requires_dataset_identity_columns(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime_path = Path(tempdir) / "r_runtime.tsv"
            runtime_path.write_text(
                "cpus\trepetition\tstep\telapsed_seconds\ttimer_runtime\t"
                "timer_runtime_version\ttimer_engine\ttimer_engine_version\t"
                "timer_host_system\ttimer_host_machine\n"
                "1\t1\tdeseq_and_results\t7\tR\t4.5.3\tDESeq2\t1.50.2\tDarwin\tarm64\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "timer_dataset"):
                COMPARATOR.read_r_runtime(runtime_path, 1, self.benchmark_timings())

    def test_external_r_runtime_rejects_mismatched_benchmark_identity(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime_path = Path(tempdir) / "r_runtime.tsv"
            valid = {
                "cpus": [1],
                "repetition": [1],
                "step": ["deseq_and_results"],
                "elapsed_seconds": [7.0],
                "timer_runtime": ["R"],
                "timer_runtime_version": ["4.5.3"],
                "timer_engine": ["DESeq2"],
                "timer_engine_version": ["1.50.2"],
                "timer_host_system": ["Darwin"],
                "timer_host_machine": ["arm64"],
                "timer_dataset": ["Pasilla"],
                "timer_genes": [8148],
                "timer_samples": [7],
                "timer_design": ["~ 0 + condition"],
                "timer_contrast": ["condition_treated_vs_untreated"],
            }
            mismatches = {
                "timer_dataset": "Pickrell",
                "timer_genes": 12531,
                "timer_samples": 69,
                "timer_design": "~ 0 + gender",
                "timer_contrast": "gender_male_vs_female",
            }
            for column, value in mismatches.items():
                with self.subTest(column=column):
                    runtime = pd.DataFrame(valid)
                    runtime.loc[0, column] = value
                    runtime.to_csv(runtime_path, sep="\t", index=False)
                    with self.assertRaisesRegex(ValueError, f"{column} does not match"):
                        COMPARATOR.read_r_runtime(runtime_path, 1, self.benchmark_timings())

    def test_numeric_cli_overrides_beat_benchmarks_without_reading_fallbacks(self):
        args = self.args(
            pydeseq2_runtime_seconds=2.5,
            pydeseq2_runtime="missing-runtime.tsv",
            r_task_seconds=3.5,
            pydeseq2_task_seconds=4.5,
            pydeseq2_session_info="missing-session-info.json",
            nextflow_trace="missing-trace.tsv",
        )

        resolved = COMPARATOR.resolve_runtime_inputs(args, self.benchmark_timings())

        self.assertEqual(resolved["pydeseq2_core_runtime_seconds"], 2.5)
        self.assertEqual(resolved["pydeseq2_core_runtime_source"], "cli_override")
        self.assertEqual(resolved["r_nfcore_task_seconds"], 3.5)
        self.assertEqual(resolved["r_nfcore_task_source"], "cli_override")
        self.assertEqual(resolved["pydeseq2_nfcore_task_seconds"], 4.5)
        self.assertEqual(resolved["pydeseq2_nfcore_task_source"], "cli_override")

    def test_invalid_numeric_cli_overrides_are_rejected(self):
        cases = (
            ({"pydeseq2_runtime_seconds": -1.0}, "--pydeseq2-runtime-seconds"),
            ({"pydeseq2_runtime_seconds": float("inf")}, "--pydeseq2-runtime-seconds"),
            ({"r_task_seconds": 0.0}, "--r-task-seconds"),
            ({"pydeseq2_task_seconds": float("nan")}, "--pydeseq2-task-seconds"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    COMPARATOR.resolve_runtime_inputs(self.args(**overrides), self.benchmark_timings())

    def test_session_cpu_count_must_match_selected_benchmark_cpu(self):
        with tempfile.TemporaryDirectory() as tempdir:
            session_path = Path(tempdir) / "session.log"
            session_path.write_text(
                '{"options": {"cores": 4, "vs_method": "vst", '
                '"runtime_seconds": {"pydeseq2_core_deseq2_and_stats": 6.5}}}\n',
                encoding="utf-8",
            )
            timings = self.benchmark_timings().assign(core_seconds=[101.0, None])

            with self.assertRaisesRegex(ValueError, "session CPU count does not match"):
                COMPARATOR.resolve_runtime_inputs(
                    self.args(cpus=1, pydeseq2_session_info=str(session_path)),
                    timings,
                )

            resolved = COMPARATOR.resolve_runtime_inputs(
                self.args(cpus=4, pydeseq2_session_info=str(session_path)),
                timings,
            )

        self.assertEqual(resolved["pydeseq2_core_runtime_seconds"], 6.5)
        self.assertEqual(resolved["pydeseq2_cores"], 4)

    def test_current_runtime_tsv_beats_session_fallback(self):
        with tempfile.TemporaryDirectory() as tempdir:
            runtime_path = Path(tempdir) / "comparison.pydeseq2.runtime.tsv"
            runtime_path.write_text(
                "step\telapsed_seconds\tcall_count\ttiming_semantics\n"
                "core_dds_deseq2\t5.25\t1\tinclusive\n"
                "stats_summary\t1.25\t1\tinclusive\n"
                "vst_dds_vst\t2.0\t1\tinclusive\n",
                encoding="utf-8",
            )
            args = self.args(
                pydeseq2_runtime=str(runtime_path),
                pydeseq2_session_info="missing-session-info.json",
            )

            resolved = COMPARATOR.resolve_runtime_inputs(args, pd.DataFrame())

        self.assertEqual(resolved["pydeseq2_core_runtime_seconds"], 6.5)
        self.assertEqual(
            resolved["pydeseq2_core_runtime_source"],
            "pydeseq2_runtime_tsv:core_dds_deseq2+stats_summary",
        )

    def test_session_and_trace_are_used_only_when_benchmark_values_are_absent(self):
        with tempfile.TemporaryDirectory() as tempdir:
            session_path = Path(tempdir) / "session.log"
            session_path.write_text(
                '{"options": {"cores": 4, "vs_method": "vst", '
                '"runtime_seconds": {"pydeseq2_core_deseq2_and_stats": 6.5}}}\n',
                encoding="utf-8",
            )
            trace_path = Path(tempdir) / "trace.tsv"
            trace_path.write_text(
                "process\trealtime\n"
                "WORKFLOW:DESEQ2_DIFFERENTIAL\t1.5s\n"
                "WORKFLOW:PYDESEQ2_DIFFERENTIAL\t2500ms\n",
                encoding="utf-8",
            )
            benchmark_args = self.args(
                cpus=None,
                pydeseq2_runtime="missing-runtime.tsv",
                pydeseq2_session_info="missing-session-info.json",
                nextflow_trace="missing-trace.tsv",
            )
            fallback_args = self.args(
                cpus=None,
                pydeseq2_session_info=str(session_path),
                nextflow_trace=str(trace_path),
            )

            benchmark_resolved = COMPARATOR.resolve_runtime_inputs(benchmark_args, self.benchmark_timings())
            fallback_timings = self.benchmark_timings().assign(
                core_seconds=[None, None],
                wall_seconds=[None, None],
            )
            resolved = COMPARATOR.resolve_runtime_inputs(fallback_args, fallback_timings)

        self.assertEqual(benchmark_resolved["r_core_runtime_seconds"], 101.0)
        self.assertEqual(benchmark_resolved["r_core_runtime_source"], "benchmark_timings")
        self.assertEqual(benchmark_resolved["pydeseq2_core_runtime_seconds"], 202.0)
        self.assertEqual(benchmark_resolved["pydeseq2_core_runtime_source"], "benchmark_timings")
        self.assertEqual(benchmark_resolved["r_nfcore_task_seconds"], 303.0)
        self.assertEqual(benchmark_resolved["r_nfcore_task_source"], "benchmark_timings")
        self.assertEqual(benchmark_resolved["pydeseq2_nfcore_task_seconds"], 404.0)
        self.assertEqual(benchmark_resolved["pydeseq2_nfcore_task_source"], "benchmark_timings")
        self.assertEqual(resolved["pydeseq2_core_runtime_seconds"], 6.5)
        self.assertEqual(resolved["pydeseq2_core_runtime_source"], "pydeseq2_session_info")
        self.assertEqual(resolved["pydeseq2_cores"], 4)
        self.assertEqual(resolved["pydeseq2_vs_method"], "vst")
        self.assertEqual(resolved["r_nfcore_task_seconds"], 1.5)
        self.assertEqual(resolved["r_nfcore_task_source"], "nextflow_trace")
        self.assertEqual(resolved["pydeseq2_nfcore_task_seconds"], 2.5)
        self.assertEqual(resolved["pydeseq2_nfcore_task_source"], "nextflow_trace")


class TransformComparisonTests(unittest.TestCase):
    def test_matrix_labels_are_aligned_before_metrics_and_geometry(self):
        r_matrix = pd.DataFrame(
            {
                "s2": [2.0, 5.0, 1.0, 7.0],
                "s1": [1.0, 4.0, 0.0, 2.0],
                "s3": [4.0, 9.0, 3.0, 8.0],
            },
            index=["g2", "g1", "g4", "g3"],
        )
        py_matrix = r_matrix.loc[["g4", "g3", "g1", "g2"], ["s3", "s1", "s2"]]

        r_aligned, py_aligned, counts = COMPARATOR.align_labelled_matrices(r_matrix, py_matrix)
        metrics, per_sample = COMPARATOR.compare_aligned_matrices("vst", r_aligned, py_aligned, counts)
        geometry, coordinates = COMPARATOR.compare_vst_geometry(r_aligned, py_aligned)

        self.assertEqual(r_aligned.index.tolist(), ["g1", "g2", "g3", "g4"])
        self.assertEqual(r_aligned.columns.tolist(), ["s1", "s2", "s3"])
        pd.testing.assert_frame_equal(r_aligned, py_aligned)
        self.assertAlmostEqual(metrics["vst_flattened_pearson"], 1.0)
        self.assertEqual(metrics["vst_abs_diff_max"], 0.0)
        self.assertEqual(per_sample["sample"].tolist(), ["s1", "s2", "s3"])
        self.assertAlmostEqual(geometry["vst_sample_distance_pearson"], 1.0)
        self.assertAlmostEqual(geometry["vst_pca_procrustes_disparity"], 0.0)
        self.assertEqual(coordinates["sample"].tolist(), ["s1", "s2", "s3"])

    def test_transform_paths_must_be_supplied_in_pairs(self):
        args = Namespace(
            r_size_factors="r.tsv",
            pydeseq2_size_factors=None,
            r_normalised_counts=None,
            pydeseq2_normalised_counts=None,
            r_vst_counts=None,
            pydeseq2_vst_counts=None,
        )

        with self.assertRaisesRegex(ValueError, "Both R and PyDESeq2 size factors"):
            COMPARATOR.validate_paired_paths(args)


if __name__ == "__main__":
    unittest.main()
