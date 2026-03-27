# Fixes Backlog

This file tracks the audit findings turned into concrete fixes.

## Priority

- `P1` Make the energy run window match the actual measured workload.
- `P1` Compute efficiency from the same long-run execution used for power.
- `P1` Improve metadata traceability for each energy case.
- `P2` Remove alignment fallbacks that can hide timestamp errors.
- `P3` Fix smaller CLI, validation, and legacy-documentation inconsistencies.

## Findings

| Status | Priority | Area | Finding | Planned action |
| --- | --- | --- | --- | --- |
| Implemented | P1 | Energy methodology | Benchmark-delimited timestamps still include pre-measurement estimation and outer host overhead. | `run_start_utc`, `run_end_utc`, and `wall_ms` are now captured inside the measured long-run section of each benchmark. |
| Implemented | P1 | Energy summary | `Efficiency (unit/W)` is computed with baseline perf instead of the perf from the same energy run. | `Perf` and efficiency now use the long-run execution; baseline perf is retained only as audit metadata. |
| Implemented | P1 | Traceability | `_meta.csv` does not store the exact campaign `case_key`, only the generic mode. | Added `--energy-case-key` support in `bench`, passed it from `run_campaign.sh`, and validated it in the summary step. |
| Implemented | P2 | Alignment | The power-summary code silently falls back to a wide timestamp slack when the run window does not overlap the log. | Switched to strict run-window clipping so bad alignment now fails loudly. |
| Implemented | P3 | CLI | Energy-mode error text still mentions only `bw/compute/gemm`, even though `fft` is supported. | Updated the user-facing message. |
| Implemented | P3 | FFT robustness | `fft_bench` can emit invalid metrics on pathological zero-time cases. | Added explicit guards for zero-time / zero-call metric generation. |
| Implemented | P3 | Validation | `validate_compare.py` still requires an exact column set for `summary_compare.csv`. | Relaxed validation to required-column presence. |
| Implemented | P3 | Legacy docs/scripts | Archived energy script still documents the pre-hardening methodology and can confuse future use. | Marked the archive script as legacy and documented the active replacement in the README. |

## Verification Plan

- Run `python -m py_compile` on updated Python scripts.
- Review the diff for metadata schema changes and runner compatibility.
- On the CUDA target machine, rebuild and run a short campaign to confirm:
  - `measured_work_ms` stays close to the timestamp-delimited interval.
  - `Run duration (ms)` reflects the measured long run, not the estimate stage.
  - `Perf` and `Efficiency (unit/W)` come from the same energy execution.
