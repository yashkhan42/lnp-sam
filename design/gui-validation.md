# GUI implementation validation

Recorded 2026-10-11 on the development Mac with Python 3.11.16 and PySide6
6.12.0. This records the first implementation; the complete app acceptance
checklist in §12.15 remains open.

## Automated and rendered checks

Final automated run: **223 passed**, with 56 dependency warnings, in 18.12 s.
Command: `QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/private/tmp/lnpsam-mpl .venv/bin/python -m pytest -q`.

- Window construction with torch, SAM, EasyOCR and Matplotlib imports forbidden;
  missing local config/model does not prevent switching to Remote server.
- Add → Run → completed overlay/summary/histogram/table using the real pipeline
  with faked SAM and OCR. Changing inputs afterward preserves completed results.
- Stage-boundary cancellation, no duplicate runs, honest Cancelling state, one
  MPS fallback, and no retry after cancellation or a failing callback.
- GUI remote preflight and upload of the prepared input snapshot; live localhost
  HTTP integration, authentication, queueing, cancellation and result download.
- Raw numeric sorting, missing/boolean values, shared frozen ID column, saved
  statistics, correct in-statistics histogram population and explicit nm/px units.
- Lazy image/ID/chart loading, stale handoff discard, decoded cache accounting,
  and unchanged source transform/center across differently sized overlay legends.
- Export/analysis lifecycle isolation and safe late connection-test handoff after
  a settings dialog is closed.
- Rendered empty, calibrated and uncalibrated result screens at 1280×840 and
  1024×720. Run/Cancel stays reachable; Distribution scrolls/stacks at the minimum
  size. At default size the full chart and summary fit without scrolling.
- Module compilation, command-line help and whitespace checks.

## Initial resource observation

These measurements used Qt's **offscreen** platform with no results loaded and
no science libraries. They do not measure native display/compositor performance
or establish the native Mac/workstation acceptance targets.

| Observation | Value | Method |
|---|---:|---|
| Initial show + event processing | 1.320 s | `time.perf_counter`, including imports, QApplication, theme and window construction |
| Python baseline peak RSS | 16.97 MiB | macOS `resource.getrusage(RUSAGE_SELF).ru_maxrss` before Qt imports |
| GUI process peak RSS | 89.00 MiB | Same high-water measurement after first event processing |
| Incremental peak RSS | 72.03 MiB | Difference between those peaks; not current resident memory |
| Settled idle observation | 60.484 s | One-shot test harness, sampled after startup setup check |
| Idle CPU, one core | 0.003% | Delta `time.process_time` divided by elapsed wall time |

Production code has no recurring idle repaint/poll timer. Remote polls happen
only within an active remote worker. The display cache has a 32 MiB RGBA budget
and reserves another 32 MiB for the displayed pixmap; tests exercise eviction
and reduced display previews without modifying scientific files.

## Still to validate

- Real SAM runs and equality with CLI outputs on the three golden panels; the
  checked-in config has a placeholder checkpoint hash and its model is absent.
- Native-screen startup/idle/RSS and analysis wall-time overhead on the Mac and
  a CUDA workstation, using the same warm workload as the CLI.
- Real remote GPU/HPC processing, including the site's SSH/Slurm setup.
- 2× display scaling, increased OS text scale, keyboard/focus and assistive
  technology review across native dialogs and popup menus.
- Actual rendered contrast and all runtime failure/setup combinations on both
  operating systems.
