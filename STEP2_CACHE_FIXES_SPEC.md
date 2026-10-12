# Step-2 cache fixes — implementation spec

**Scope:** `lnpsam/proposals.py`, `lnpsam/run.py`, `lnpsam/__init__.py`, `config.yaml`, `tests/test_proposals.py`, `tests/test_run.py`, the two spec docs.
**Not in scope:** anything that changes what Step 2 produces. Same inputs must give the same nodes before and after.

> The checkpoint hash in §2.1 comes from the HPC copy of the model:
> `sha256sum /lustre/work/tkhanyas/weights/sam_vit_b_01ec64.pth`

---

## 0. Why

| # | Problem | Where | Effect |
|---|---|---|---|
| F1 | The cache key includes machine-specific values: checkpoint absolute path + `mtime_ns` (`_checkpoint_identity`), the `sam.checkpoint` path string (inside `sam_config`), the device, the installed segment-anything version, and the raw bytes of `proposals.py` (`implementation_sha256`). | `proposal_cache_key` in `proposals.py` | The same Step-2 work done on an HPC GPU node, an HPC CPU node, the Mac or a future remote server gets a different key, so it is never reused. Blocks the GUI's remote-compute mode. |
| F2 | `--proposals-from` loads nodes without checking the key, and takes `sorted(hits)[0]` when several runs contain the panel. | `run_cleanup_stage`, `find_proposal_nodes` in `run.py` | Steps 3–6 can silently run on proposals made from a different Step-1 crop/exclusion or different SAM thresholds, or from the wrong run. |
| F3 | `__meta.json` has no timing and no environment info. | `save_proposal_cache` | No record of how long Step 2 took or on what hardware; an HPC CPU node and a laptop CPU both just say `"cpu"`. |

## 1. Ground rules

1. As of 2026-10-10, `main` has **uncommitted changes** in `LNP-SAM_MVP_SPEC.md`, `config.yaml`, `lnpsam/graph.py`, `lnpsam/overlay.py`, `lnpsam/resolve.py`, `tests/test_resolve.py`, `tests/test_run.py`. Commit them separately first. Never discard them or fold them into this work's commits.
2. Work on a new branch `step2-cache-portability`; merge only after review.
3. Run `pytest` before changing anything and record the result. Everything that passed must still pass at the end.
4. Step-2 node output must not change (verify per §7).
5. No new third-party dependencies. Stdlib only (`hashlib`, `time`, `platform`, `subprocess`, `datetime`).
6. Leave `select_device` and the `--device` flag unchanged.
7. Tests must pass on a Mac with no SAM checkpoint and no GPU (fakes, as the existing tests do).
8. Commits: one per fix (F1, F2, F3) plus one for docs.

---

## 2. Fix 1 — portable cache key

### 2.1 Config

Add under `sam:` in `config.yaml`:

```yaml
  checkpoint_sha256: <PASTE>   # sha256 of sam_vit_b_01ec64.pth; identifies the model in the cache key
```

The value must be computed from the HPC copy of the checkpoint (see the note at the top), not copied from elsewhere.

`sam.checkpoint` (the path) stays. It says where *this* machine finds the file and may differ per machine.

### 2.2 Key inputs

| Input | In key? | Note |
|---|---|---|
| `gray` (as float32), `sam_image`, `exclusion` (uint8), `content_mask` (uint8) | Yes | Same casts as today |
| `content_box`, `crop_offset` | Yes | Unchanged |
| `sam:` params **except** `checkpoint` and `checkpoint_sha256` | Yes | |
| `checkpoint_sha256` | Yes | Replaces path + size + mtime |
| `STEP2_VERSION` constant | Yes | Replaces sha256 of `proposals.py` |
| Checkpoint path | **No** | Metadata only |
| Device | **No** | Metadata only |
| torch / segment-anything versions | **No** | Metadata only. A machine without SAM installed (Mac with `--proposals-from`, future GUI client) must compute the same key. |

### 2.3 Code changes in `proposals.py`

- Add module constants:
  ```python
  # Bump when Step-2 output can change for the same inputs (SAM call, post-processing,
  # features, RLE encoding). test_step2_output_fingerprint fails if you forget.
  STEP2_VERSION = 1
  _KEY_SCHEMA = 2          # legacy caches have no key_schema
  _UNKEYED_SAM_FIELDS = frozenset({"checkpoint", "checkpoint_sha256"})
  ```
- `_require_checkpoint_sha256(sam_config) -> str`: return the value lowercased. Raise `ValueError` if missing or not exactly 64 hex characters. The message says how to compute it (`sha256sum <file>` on Linux, `shasum -a 256 <file>` on macOS).
- `proposal_input_digests(gray, cleanup_result, sam_config) -> dict[str, str]` with exactly these keys:
  `gray`, `sam_image`, `exclusion`, `content_mask`, `geometry`, `step2_params`, `checkpoint_sha256`, `step2_version`.
  - Array digests: sha256 hex over name + shape + dtype + bytes (one digest per array; reuse the `_hash_array` logic).
  - `geometry`: sha256 of canonical JSON of `{"content_box": [...], "crop_offset": [...]}`.
  - `step2_params`: sha256 of canonical JSON of the `sam:` dict minus `_UNKEYED_SAM_FIELDS`.
  - `checkpoint_sha256`: the validated value. `step2_version`: `str(STEP2_VERSION)`.
  - Canonical JSON everywhere: `json.dumps(obj, sort_keys=True, separators=(",", ":"))`.
- `proposal_cache_key(gray, cleanup_result, sam_config) -> str`: sha256 of canonical JSON of `{"key_schema": _KEY_SCHEMA, **digests}`.
  **Signature change:** drop `checkpoint` and `device`; return the key only (no tuple). Update every caller.
- Delete `_checkpoint_identity`.
- Add `verify_checkpoint(path, expected_sha256)`: streaming sha256 in 1 MiB chunks; on mismatch raise `ValueError` naming the path and both hashes.
- `load_sam_model(checkpoint, device, expected_sha256)`: call `verify_checkpoint` before loading. Costs ~1–2 s, and only when SAM actually loads (already lazy in `run.py`).
- `run_proposals_cached(...)`: drop the `checkpoint` parameter; keep `device` (metadata only).
- In `run.py`, `get_mask_generator()` passes `expected_sha256=cfg["sam"]["checkpoint_sha256"]`.

### 2.4 Metadata written by `save_proposal_cache`

```json
{
  "panel_id": "…",
  "key_schema": 2,
  "cache_key": "…",
  "step2_version": 1,
  "input_digests": {"gray": "…", "sam_image": "…", "exclusion": "…", "content_mask": "…",
                    "geometry": "…", "step2_params": "…", "checkpoint_sha256": "…", "step2_version": "1"},
  "checkpoint_path": "/path/as/resolved/on/this/machine",
  "device": "cuda",
  "n_raw_masks": 76,
  "n_nodes": 84,
  "created_utc": "2026-10-11T03:20:00Z",
  "provenance": {"…": "see §4.1"},
  "timing": {"…": "see §4.2"}
}
```

`load_proposal_cache` keeps its current checks (key equal, columns, row count). Legacy caches (no `key_schema`) miss and recompute. That is intended.

---

## 3. Fix 2 — validate `--proposals-from`

### 3.1 Behaviour, per panel, after the local Step 1

1. Compute `expected_key` and `expected_digests` from the local Step-1 result and local `config.yaml`. No SAM needed.
2. Candidates: every `proposals/<panel_id>__nodes.parquet` under DIR (same glob as today; still excludes `measure/`).
3. Read each candidate's sibling `__meta.json`. Missing or unreadable → `{}`.
4. Decide:

| Situation | Default | With `--allow-stale-proposals` |
|---|---|---|
| ≥1 candidate has `cache_key == expected_key` | Use it. If several, newest `created_utc`; print which. `validated = True` | Same |
| No candidates | `FileNotFoundError` (as today) | Same |
| Candidates exist, none match | Raise `StaleProposalsError` (§3.2) | Exactly 1 candidate → use it, `validated = False`, print a `WARNING:` line for the panel. More than 1 → raise `StaleProposalsError` asking to point DIR at a single run |

### 3.2 `StaleProposalsError`

Define `class StaleProposalsError(RuntimeError)` in `proposals.py`. The message names the panel, then lists each candidate path with its reason:

- `no __meta.json`
- `made by an older LNP-SAM (no key_schema); rerun Step 2`
- `differs in: exclusion, step2_params`: the `input_digests` entries that differ from `expected_digests`

It ends with: `Rerun Step 2 for this panel, or pass --allow-stale-proposals to use it anyway.`

`main()` catches `StaleProposalsError`, prints the message to stderr and exits with code 2 (no traceback).

### 3.3 CLI and functions in `run.py`

- Add `--allow-stale-proposals` (`store_true`). Using it without `--proposals-from` → `parser.error(...)`.
- `run_cleanup_stage(..., allow_stale_proposals=False)`.
- Replace `find_proposal_nodes` with `select_proposal_nodes(proposals_from, panel_id, expected_key, expected_digests, allow_stale) -> (nodes_path, meta, validated)`.
- Update the module docstring and the `--proposals-from` help text: nodes are reused only when their key matches.

### 3.4 `proposals_all.csv`

Add columns:

| Column | Value |
|---|---|
| `proposals_validated` | True for local Step-2 runs and validated reuse; False for stale reuse |
| `proposals_source` | Path of the nodes file used |
| `step2_seconds` | `timing.total_s` from the meta; empty if absent |

For `--proposals-from` rows, fill `device` from the source meta (today it is empty).

---

## 4. Fix 3 — provenance and timing

### 4.1 `provenance`

Helper `_collect_provenance(device) -> dict`. **Must never raise**: any failing field becomes `null`.

| Field | Source |
|---|---|
| `device` | Resolved device |
| `accelerator` | `torch.cuda.get_device_name(idx)` for cuda; `"Apple MPS"` for mps; `platform.processor() or platform.machine()` for cpu |
| `cpu_count` | `os.cpu_count()` |
| `torch_threads` | `torch.get_num_threads()` |
| `torch` | `torch.__version__` |
| `segment_anything` | `importlib.metadata.version("segment-anything")`, else `"missing"` |
| `python` | `platform.python_version()` |
| `os` | `platform.platform()` |
| `lnpsam_version` | `lnpsam.__version__`. Add `__version__ = "0.1.0"` to `lnpsam/__init__.py` if absent |
| `git_commit` | `git rev-parse --short HEAD` in the package directory, 2 s timeout; append `+dirty` if `git status --porcelain` is non-empty; `null` if not a git checkout (the HPC copy may have no `.git`) |

No hostname, no username.

### 4.2 `timing`

Seconds via `time.perf_counter()`, rounded to 3 decimals. Recorded only on a cache miss. On a hit the stored meta is returned unchanged, since it describes the run that made the proposals.

| Field | Measures |
|---|---|
| `model_load_s` | The `mask_generator_factory()` call. ~0 after the first miss because the factory caches the model |
| `generate_s` | `mask_generator.generate(...)` |
| `postprocess_s` | `postprocess_proposals(...)` |
| `total_s` | Sum of the three |

---

## 5. Tests

### 5.1 `tests/test_proposals.py`

Add a valid `checkpoint_sha256` (e.g. `"0" * 64`) to `SAM_CFG`. Update the two existing cache tests to the new signatures.

| New test | Asserts |
|---|---|
| Key ignores the checkpoint path | Same key when `sam_config["checkpoint"]` differs |
| Key tracks what matters | Key changes when `checkpoint_sha256` changes; when each of `points_per_side`, `pred_iou_thresh`, `stability_score_thresh`, `box_nms_thresh`, `min_piece_px` changes (parametrize); when `STEP2_VERSION` is monkeypatched; when `crop_offset` changes |
| sha validation | Missing, 63-char and non-hex values → `ValueError`. Uppercase accepted and gives the same key as lowercase |
| `verify_checkpoint` | Matching file passes; mismatch raises `ValueError` whose message contains both hashes |
| Metadata on a miss | `key_schema`, `step2_version`, `input_digests` (8 keys), `device`, `provenance` (every §4.1 field present), `timing` (4 non-negative numbers), `created_utc` |
| Metadata on a hit | Returned meta equals the stored meta |
| Provenance never raises | Monkeypatch `subprocess.run` to raise → `git_commit` is `None`, no exception |
| `test_step2_output_fingerprint` | Run `postprocess_proposals` on a fixed toy panel with fixed fake annotations; map nodes through `_node_to_cache_record`; round floats to 6 dp; sha256 of canonical JSON equals a pinned constant. Failure message: *"Step-2 output changed. If intended, bump STEP2_VERSION in proposals.py and update this constant."* Pin the constant from the **pre-change** code so the test also proves outputs did not change. |

### 5.2 `tests/test_run.py`

The first integration test (~line 37) builds a config from `config.yaml`; make sure it carries a valid `checkpoint_sha256`.

Rewrite `test_proposals_from_reuses_nodes_without_sam` as a realistic two-run test:

1. **"HPC" run:** `run_cleanup_stage(..., run_step2=True, device="cuda", out_dir=<tmp>/hpc_run)`, with `select_device` patched to return `"cuda"`, `load_sam_model` patched to return a dummy, `make_mask_generator` patched to return the fake generator. Config `sam.checkpoint` = `<tmp>/A/sam.pth`.
2. **"Mac" run:** `main([..., "--proposals-from", <tmp>/hpc_run])` with config `sam.checkpoint` = `<tmp>/B/sam.pth` (different path), and `load_sam_model` / `select_device` patched to raise.

Assert: particles are written; `proposals_validated` is True; `device == "cuda"`; `cache_key` equals the HPC meta's.

| New test | Setup | Expect |
|---|---|---|
| Refuses changed params | Mac config `pred_iou_thresh` differs | Exit code 2; stderr has the panel id and `step2_params` |
| Refuses changed Step 1 | Patch Step 1 on the Mac run to flip one exclusion pixel | Message contains `exclusion` |
| Refuses legacy cache | Candidate meta without `key_schema` | Message contains `older LNP-SAM` |
| Picks the matching run, not alphabetical first | `aaa_stale/` (mismatched) and `zzz_good/` (matching) under DIR | Uses `zzz_good` |
| Allow-stale, one candidate | One mismatched candidate + flag | Runs; `proposals_validated` False; `WARNING:` in stdout |
| Allow-stale, two candidates | Two mismatched + flag | Exit code 2; message asks to narrow DIR |
| Flag misuse | `--allow-stale-proposals` without `--proposals-from` | argparse error |

---

## 6. Spec docs

- `LNP-SAM_MVP_SPEC.md` §4.2 (the `--proposals-from` sentence, ~line 85): replace "loaded as-is" with: nodes are reused only if their cache key matches what the local Step 1 and `config.yaml` would produce; otherwise the run stops (override: `--allow-stale-proposals`).
- §4.5 config block: add `checkpoint_sha256` with its comment.
- §6: add a short **6.4 Step-2 cache** subsection: the key-inputs table (§2.2 here), the metadata fields (§2.4, §4), and the `STEP2_VERSION` bump rule.
- `LNP-SAM_MVP_SPEC_AMENDMENTS.md`: append a section after §13 in the same style (problem table, then choice table with a "Why" column), titled **"14. Step-2 cache portability and validated reuse (changes §4.2, §4.5, §6)"**, using the next revision label after the spec's current one. State that it supersedes §13's note "`proposals.py` untouched, so the HPC proposal cache stays valid": after this change every existing cache misses once.

---

## 7. Done when

- [ ] Baseline tests still pass; all new tests pass; `pytest` is green on the Mac with no checkpoint and no GPU.
- [ ] `test_step2_output_fingerprint` passes with the constant pinned from the pre-change code (proves Step-2 output is unchanged).
- [ ] `grep -rn "mtime_ns\|_checkpoint_identity\|implementation_sha256" lnpsam/` returns nothing.
- [ ] Both spec docs updated.
- [ ] Four commits on `step2-cache-portability` (F1, F2, F3, docs); not merged.

## 8. After merge (manual)

1. Re-upload the repo to the HPC as usual.
2. Rerun Step 2 for all panels on the HPC. Every cache misses once, by design.
3. On the Mac, run with `--proposals-from <hpc_run>`. Every panel should report validated. If one is refused, the message names what differs. Most likely the Mac's Step 1 doesn't match the HPC's for that panel (e.g. an EasyOCR difference), which is worth knowing.
4. Old run folders (`runs/mvp_v1`, `runs/mvp_v2`, …) stay readable as results, but `--proposals-from` on them refuses (legacy) unless you pass `--allow-stale-proposals`.
