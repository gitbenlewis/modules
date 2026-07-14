import importlib.util
import math
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "summarise_scaling.py"
SPEC = importlib.util.spec_from_file_location("summarise_scaling", SCRIPT)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)
RUNNER_SCRIPT = Path(__file__).resolve().parents[1] / "run_benchmark_repetitions.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("run_benchmark_repetitions", RUNNER_SCRIPT)
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)


class RobustSpreadTests(unittest.TestCase):
    def test_iqr_and_unscaled_mad_resist_an_outlier_and_drop_invalid_values(self):
        values = pd.Series([1, 2, 3, 4, 100, None, "invalid"])

        self.assertAlmostEqual(SUMMARY.interquartile_range(values), 2.0)
        self.assertAlmostEqual(SUMMARY.median_absolute_deviation(values), 1.0)

    def test_crossover_requires_each_implementation_to_win(self):
        self.assertFalse(SUMMARY.performance_crossover(["tie", "r"]))
        self.assertFalse(SUMMARY.performance_crossover(["tie", "pydeseq2"]))
        self.assertTrue(SUMMARY.performance_crossover(["r", "tie", "pydeseq2"]))


class CompleteMatrixTests(unittest.TestCase):
    def setUp(self):
        self.timings = pd.DataFrame(
            [
                {
                    "dataset": "tiny",
                    "implementation": implementation,
                    "cpus": cpus,
                    "repetition": repetition,
                    "genes": 10,
                    "samples": 4,
                    "design": "~ group",
                    "contrast": "group_treated_vs_control",
                    "task_seconds": float(cpus + repetition),
                    "workflow_seconds": float(cpus + repetition + 1),
                    "task_timing_source": "nextflow_trace",
                    "workflow_timing_source": "python_perf_counter",
                    "core_seconds": 1.0,
                    "vst_seconds": 0.5,
                    "peak_rss_bytes": None,
                    "output_bytes": 100,
                    "result_sha256": f"result-{implementation}",
                    "data_outputs_sha256": f"outputs-{implementation}",
                    "observed_cpus": cpus,
                    "native_math_threads_per_worker": 1,
                    "profile": "local",
                    "r_version": "4.5.3" if implementation == "r" else None,
                    "deseq2_version": "1.50.2" if implementation == "r" else None,
                    "python_version": "3.11" if implementation == "pydeseq2" else None,
                    "pydeseq2_version": "0.5" if implementation == "pydeseq2" else None,
                    "nextflow_version": "25.10",
                    "host_system": "test",
                    "host_machine": "test",
                    "output_dir": "unused",
                }
                for implementation in ("r", "pydeseq2")
                for cpus in (1, 4)
                for repetition in (1, 2)
            ]
        )

    def r_runtime(self):
        return pd.DataFrame(
            [
                {
                    "cpus": cpus,
                    "repetition": repetition,
                    "step": step,
                    "elapsed_seconds": value,
                    "timer_runtime": "R",
                    "timer_runtime_version": "4.5.3",
                    "timer_engine": "DESeq2",
                    "timer_engine_version": "1.50.2",
                    "timer_host_system": "test",
                    "timer_host_machine": "test",
                    "timer_dataset": "tiny",
                    "timer_genes": 10,
                    "timer_samples": 4,
                    "timer_design": "~ group",
                    "timer_contrast": "group_treated_vs_control",
                }
                for cpus in (1, 4)
                for repetition in (1, 2)
                for step, value in (
                    ("deseq_and_results", float(cpus * 10 + repetition)),
                    ("vst", float(cpus + repetition / 10)),
                )
            ]
        )

    def test_complete_matrix_is_accepted(self):
        datasets, cpus, repetitions = SUMMARY.validate_complete_matrix(self.timings.copy())

        self.assertEqual(datasets, ["tiny"])
        self.assertEqual(cpus, [1, 4])
        self.assertEqual(repetitions, [1, 2])

    def test_missing_cell_is_rejected(self):
        timings = self.timings.drop(index=0)

        with self.assertRaisesRegex(ValueError, "Incomplete benchmark matrix"):
            SUMMARY.validate_complete_matrix(timings)

    def test_duplicate_cell_is_rejected(self):
        timings = pd.concat([self.timings, self.timings.iloc[[0]]], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "Duplicate benchmark cells"):
            SUMMARY.validate_complete_matrix(timings)

    def test_non_finite_task_duration_is_rejected(self):
        for value in (math.nan, math.inf):
            with self.subTest(value=value):
                timings = self.timings.copy()
                timings.loc[0, "task_seconds"] = value

                with self.assertRaisesRegex(ValueError, "task_seconds must contain a finite positive value"):
                    SUMMARY.validate_complete_matrix(timings)

    def test_r_core_override_is_applied_and_effective_timings_are_auditable(self):
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            timings_path = tmpdir / "timings.tsv"
            runtime_path = tmpdir / "r_runtime.tsv"
            outdir = tmpdir / "report"
            self.timings.to_csv(timings_path, sep="\t", index=False)
            self.r_runtime().to_csv(runtime_path, sep="\t", index=False)
            phase_runs = pd.DataFrame(
                [
                    {
                        "dataset": "tiny",
                        "genes": 10,
                        "samples": 4,
                        "cpus": cpus,
                        "repetition": repetition,
                        "step": "script_total",
                        "elapsed_seconds": 2.0,
                        "call_count": 1,
                        "timing_semantics": "inclusive_non_additive",
                        "runtime_file": "unused",
                    }
                    for cpus in (1, 4)
                    for repetition in (1, 2)
                ]
            )
            args = Namespace(
                timings=[str(timings_path)],
                outdir=outdir,
                r_core_runtime=[f"TINY={runtime_path}"],
            )

            with patch.object(SUMMARY, "parse_args", return_value=args), patch.object(
                SUMMARY,
                "collect_pydeseq2_phases",
                return_value=phase_runs,
            ):
                SUMMARY.main()

            effective = pd.read_csv(outdir / "benchmark_effective_timings.tsv", sep="\t")
            r_rows = effective.loc[effective["implementation"] == "r"].sort_values(
                ["cpus", "repetition"]
            )
            py_rows = effective.loc[effective["implementation"] == "pydeseq2"]

            self.assertFalse(any(column.startswith("_") for column in effective.columns))
            self.assertEqual(r_rows["core_seconds"].tolist(), [11.0, 12.0, 41.0, 42.0])
            self.assertEqual(r_rows["vst_seconds"].tolist(), [1.1, 1.2, 4.1, 4.2])
            self.assertEqual(
                set(r_rows["core_timing_source"]),
                {"standalone_r_reference_runtime_tsv:deseq_and_results"},
            )
            self.assertEqual(set(r_rows["core_timer_runtime"]), {"R"})
            self.assertEqual(set(r_rows["core_timer_runtime_version"]), {"4.5.3"})
            self.assertEqual(set(r_rows["core_timer_engine"]), {"DESeq2"})
            self.assertEqual(set(r_rows["core_timer_engine_version"]), {"1.50.2"})
            self.assertEqual(set(r_rows["core_timer_host_system"]), {"test"})
            self.assertEqual(set(r_rows["core_timer_host_machine"]), {"test"})
            self.assertEqual(set(r_rows["core_timer_dataset"]), {"tiny"})
            self.assertEqual(set(r_rows["core_timer_genes"]), {10})
            self.assertEqual(set(r_rows["core_timer_samples"]), {4})
            self.assertEqual(set(r_rows["core_timer_design"]), {"~ group"})
            self.assertEqual(set(r_rows["core_timer_contrast"]), {"group_treated_vs_control"})
            for column in ("core_timing_file", "vst_timing_file", "r_core_runtime_path"):
                self.assertEqual(set(r_rows[column]), {runtime_path.name})
            self.assertTrue(py_rows["core_seconds"].eq(1.0).all())
            self.assertTrue(py_rows["vst_seconds"].eq(0.5).all())
            self.assertIn(
                "performance_crossover_task_across_tested_datasets",
                pd.read_csv(outdir / "benchmark_scaling_comparison.tsv", sep="\t").columns,
            )

    def test_r_core_override_rejects_incomplete_core_rows(self):
        runtime = self.r_runtime()
        runtime = runtime.loc[
            ~(
                runtime["cpus"].eq(4)
                & runtime["repetition"].eq(2)
                & runtime["step"].eq("deseq_and_results")
            )
        ]
        with TemporaryDirectory() as tmpdir:
            runtime_path = Path(tmpdir) / "r_runtime.tsv"
            runtime.to_csv(runtime_path, sep="\t", index=False)

            with self.assertRaisesRegex(ValueError, "deseq_and_results rows do not exactly match"):
                SUMMARY.apply_r_core_runtime_overrides(
                    self.timings,
                    SUMMARY.parse_r_core_runtime_specs([f"tiny={runtime_path}"]),
                )

    def test_r_core_override_rejects_duplicate_runtime_keys(self):
        runtime = self.r_runtime()
        runtime = pd.concat([runtime, runtime.iloc[[0]]], ignore_index=True)
        with TemporaryDirectory() as tmpdir:
            runtime_path = Path(tmpdir) / "r_runtime.tsv"
            runtime.to_csv(runtime_path, sep="\t", index=False)

            with self.assertRaisesRegex(ValueError, "duplicate \\(cpus, repetition, step\\) rows"):
                SUMMARY.apply_r_core_runtime_overrides(
                    self.timings,
                    SUMMARY.parse_r_core_runtime_specs([f"tiny={runtime_path}"]),
                )

    def test_r_core_override_rejects_ambiguous_dataset_case(self):
        timings = self.timings.copy()
        r_indexes = timings.index[timings["implementation"].eq("r")]
        timings.loc[r_indexes[0], "dataset"] = "TINY"
        with TemporaryDirectory() as tmpdir:
            runtime_path = Path(tmpdir) / "r_runtime.tsv"
            self.r_runtime().to_csv(runtime_path, sep="\t", index=False)

            with self.assertRaisesRegex(ValueError, "Ambiguous case-insensitive R timing datasets"):
                SUMMARY.apply_r_core_runtime_overrides(
                    timings,
                    SUMMARY.parse_r_core_runtime_specs([f"tiny={runtime_path}"]),
                )

    def test_r_core_override_rejects_timer_metadata_mismatch(self):
        runtime = self.r_runtime()
        runtime["timer_engine_version"] = "1.49.0"
        with TemporaryDirectory() as tmpdir:
            runtime_path = Path(tmpdir) / "r_runtime.tsv"
            runtime.to_csv(runtime_path, sep="\t", index=False)

            with self.assertRaisesRegex(ValueError, "timer_engine_version does not match"):
                SUMMARY.apply_r_core_runtime_overrides(
                    self.timings,
                    SUMMARY.parse_r_core_runtime_specs([f"tiny={runtime_path}"]),
                )

    def test_r_core_override_requires_local_profile(self):
        timings = self.timings.assign(profile="docker")
        with TemporaryDirectory() as tmpdir:
            runtime_path = Path(tmpdir) / "r_runtime.tsv"
            self.r_runtime().to_csv(runtime_path, sep="\t", index=False)

            with self.assertRaisesRegex(ValueError, "matched local benchmark"):
                SUMMARY.apply_r_core_runtime_overrides(
                    timings,
                    SUMMARY.parse_r_core_runtime_specs([f"tiny={runtime_path}"]),
                )

    def test_r_core_override_rejects_partially_missing_profile(self):
        timings = self.timings.copy()
        r_index = timings.index[timings["implementation"].eq("r")][0]
        timings.loc[r_index, "profile"] = pd.NA
        with TemporaryDirectory() as tmpdir:
            runtime_path = Path(tmpdir) / "r_runtime.tsv"
            self.r_runtime().to_csv(runtime_path, sep="\t", index=False)

            with self.assertRaisesRegex(ValueError, "matched local benchmark"):
                SUMMARY.apply_r_core_runtime_overrides(
                    timings,
                    SUMMARY.parse_r_core_runtime_specs([f"tiny={runtime_path}"]),
                )

    def test_r_core_override_rejects_timer_identity_mismatch(self):
        runtime = self.r_runtime()
        runtime["timer_dataset"] = "other"
        with TemporaryDirectory() as tmpdir:
            runtime_path = Path(tmpdir) / "r_runtime.tsv"
            runtime.to_csv(runtime_path, sep="\t", index=False)

            with self.assertRaisesRegex(ValueError, "timer_dataset does not match"):
                SUMMARY.apply_r_core_runtime_overrides(
                    self.timings,
                    SUMMARY.parse_r_core_runtime_specs([f"tiny={runtime_path}"]),
                )


class RunnerSafetyTests(unittest.TestCase):
    def test_external_r_runtime_requires_one_local_environment(self):
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            runtime_path = tmpdir / "r_runtime.tsv"
            runtime_path.write_text("placeholder\n", encoding="utf-8")
            shared_bin = tmpdir / "shared" / "bin"
            other_bin = tmpdir / "other" / "bin"
            shared_bin.mkdir(parents=True)
            other_bin.mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "requires --profile local"):
                RUNNER.validate_external_r_runtime_setup(
                    "docker", runtime_path, shared_bin, shared_bin
                )
            with self.assertRaisesRegex(ValueError, "requires both --r-env-bin"):
                RUNNER.validate_external_r_runtime_setup("local", runtime_path, None, shared_bin)
            with self.assertRaisesRegex(ValueError, "resolve to the same directory"):
                RUNNER.validate_external_r_runtime_setup(
                    "local", runtime_path, shared_bin, other_bin
                )

            RUNNER.validate_external_r_runtime_setup(
                "local", runtime_path, shared_bin, shared_bin
            )

    def test_pre_existing_run_root_is_rejected_without_mutation(self):
        with TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "r_1cpu" / "run_1"
            run_root.mkdir(parents=True)
            marker = run_root / "keep.txt"
            marker.write_text("keep\n", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "Refusing to reuse"):
                RUNNER.create_run_root(run_root)

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep\n")

    def test_r_timer_metadata_must_match_module_session_and_host(self):
        metadata = {
            "timer_runtime": "R",
            "timer_runtime_version": "4.5.3",
            "timer_engine": "DESeq2",
            "timer_engine_version": "1.50.2",
            "timer_host_system": "Darwin",
            "timer_host_machine": "arm64",
        }
        versions = {"r_version": "4.5.3", "deseq2_version": "1.50.2"}

        RUNNER.validate_r_timer_metadata(metadata, versions, "Darwin", "arm64")
        with self.assertRaisesRegex(ValueError, "timer_engine_version"):
            RUNNER.validate_r_timer_metadata(
                {**metadata, "timer_engine_version": "1.49.0"},
                versions,
                "Darwin",
                "arm64",
            )

    def test_r_timer_identity_must_match_selected_input(self):
        identity = {
            "timer_dataset": "Pasilla",
            "timer_genes": "8148",
            "timer_samples": "7",
            "timer_design": "~ 0 + condition",
            "timer_contrast": "condition_treated_vs_untreated",
        }

        RUNNER.validate_r_timer_identity(
            identity,
            "pasilla",
            8148,
            7,
            "~ 0 + condition",
            "condition_treated_vs_untreated",
        )
        with self.assertRaisesRegex(ValueError, "timer_genes"):
            RUNNER.validate_r_timer_identity(
                {**identity, "timer_genes": "12531"},
                "Pasilla",
                8148,
                7,
                "~ 0 + condition",
                "condition_treated_vs_untreated",
            )


if __name__ == "__main__":
    unittest.main()
