from __future__ import annotations

import json
import unittest

from mlsys_phase2.profile import summarize_ncu_stdout
from mlsys_phase2.prompts import benchmark_prompt_json, profile_prompt_json


class PromptSummaryTests(unittest.TestCase):
    def test_summarize_ncu_stdout_filters_empty_and_zero_metrics(self) -> None:
        stdout = """Kernel Name,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed
void lora_gemm_kernel(),0,12.5,0
"""

        summary = summarize_ncu_stdout(
            stdout,
            [
                "gpu__time_duration.sum",
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                "lts__throughput.avg.pct_of_peak_sustained_elapsed",
            ],
        )

        self.assertIn("sm__throughput.avg.pct_of_peak_sustained_elapsed", summary["metrics"])
        self.assertNotIn("gpu__time_duration.sum", summary["metrics"])
        self.assertNotIn("dram__throughput.avg.pct_of_peak_sustained_elapsed", summary["metrics"])
        self.assertNotIn("lts__throughput.avg.pct_of_peak_sustained_elapsed", summary["metrics"])
        self.assertEqual(summary["kernel_count"], 1)
        self.assertEqual(summary["kernels"][0]["name"], "void lora_gemm_kernel()")

    def test_profile_prompt_json_keeps_only_compact_fields(self) -> None:
        profile_result = {
            "ok": True,
            "sampled_d": 4096,
            "error": None,
            "cases": [
                {
                    "d": 4096,
                    "ok": True,
                    "returncode": 0,
                    "requested_metrics": ["gpu__time_duration.sum"],
                    "stderr": "large stderr",
                    "summary": {
                        "kernel_count": 2,
                        "kernels": [{"name": "void my_gemm_kernel()", "count": 2}],
                        "raw_stdout_chars": 99999,
                        "metrics": {
                            "gpu__time_duration.sum": {
                                "count": 2,
                                "min": 10.0,
                                "max": 12.0,
                                "avg": 11.0,
                                "values": [10.0, 12.0],
                            },
                            "empty_metric": {
                                "count": 0,
                                "min": None,
                                "max": None,
                                "avg": None,
                                "values": [],
                            },
                            "zero_metric": {
                                "count": 2,
                                "min": 0.0,
                                "max": 0.0,
                                "avg": 0.0,
                                "values": [0.0, 0.0],
                            },
                        },
                    },
                }
            ],
        }

        text = profile_prompt_json(profile_result)
        compact = json.loads(text)

        self.assertTrue(compact["ok"])
        self.assertEqual(compact["sampled_d"], 4096)
        self.assertEqual(compact["cases"][0]["kernel_count"], 2)
        self.assertEqual(compact["cases"][0]["kernels"][0], {"name": "void my_gemm_kernel()", "count": 2})
        self.assertIn("gpu__time_duration.sum", compact["cases"][0]["metrics"])
        self.assertNotIn("empty_metric", compact["cases"][0]["metrics"])
        self.assertNotIn("zero_metric", compact["cases"][0]["metrics"])
        self.assertNotIn("values", text)
        self.assertNotIn("stderr", text)
        self.assertNotIn("requested_metrics", text)
        self.assertNotIn("raw_stdout_chars", text)

    def test_benchmark_prompt_json_drops_none_fields_and_keeps_core_info(self) -> None:
        benchmark_result = {
            "ok": False,
            "average_speedup": None,
            "error": "compile failed",
            "cases": [
                {
                    "d": 1024,
                    "correct": False,
                    "speedup": None,
                    "student_median_ms": None,
                    "torch_median_ms": 1.5,
                    "max_abs_err": 0.1,
                    "rel_l2_err": None,
                    "error": "compile failed",
                }
            ],
        }

        text = benchmark_prompt_json(benchmark_result)
        compact = json.loads(text)

        self.assertFalse(compact["ok"])
        self.assertEqual(compact["error"], "compile failed")
        self.assertEqual(compact["unique_errors"], ["compile failed"])
        self.assertEqual(compact["cases"][0]["d"], 1024)
        self.assertFalse(compact["cases"][0]["correct"])
        self.assertEqual(compact["cases"][0]["torch_median_ms"], 1.5)
        self.assertEqual(compact["cases"][0]["max_abs_err"], 0.1)
        self.assertNotIn("average_speedup", compact)
        self.assertNotIn("speedup", compact["cases"][0])
        self.assertNotIn("student_median_ms", compact["cases"][0])
        self.assertNotIn("rel_l2_err", compact["cases"][0])


if __name__ == "__main__":
    unittest.main()
