# LNP-SAM App — v1 spec

**Goal:** a simple desktop app that runs the existing LNP-SAM pipeline on images a user adds, then shows and saves the results. The pipeline runs either **on this computer** or **on a remote computer** (a lab GPU machine or an HPC node).
**Assumes:** the Step-2 cache fixes are merged.
**New code:** `lnpsam/gui.py` (the window), `lnpsam/server.py` (remote mode), `lnpsam/remote.py` (client for the server), plus one small change in `run.py`.

---

## 1. What v1 is

- A **PySide6** desktop window. Launched from source with `python -m lnpsam.gui` (no installer yet).
- **Fully automatic.** The user adds images, picks TEM/SEM, clicks Run.
- **Two compute options**, chosen in the window:
  - **This computer:** NVIDIA GPU, then Apple GPU, then CPU, chosen automatically.
  - **Remote server:** the same pipeline running on another machine (§5). The app uploads the images and downloads the results.
- Both options produce the **same results folder**, and the window displays it with the same code.

**Not in v1:** installers (.dmg/.exe), code signing, editing particles, tiling large images, DM3 files, HTTPS.

---

## 2. Window layout (two panels)

The visual and interaction contract is in **§12**. Use a dark, compact desktop
workspace with native Qt controls. Colors and dimensions are also recorded in
`design/lnpsam-ui-tokens.json`.

**Left panel: input**

| Control | Widget | Default | Notes |
|---|---|---|---|
| Images | List + "Add images…" button; drag and drop | — | PNG, JPG, TIF. Shows name and size; large images get a ⚠ icon (§7) |
| Modality | Radio: TEM / SEM | TEM | Applies to all images |
| Scale (nm per pixel) | Number field, optional | Empty = auto-detect from the scale bar | |
| Compute | Dropdown: "This computer" / "Remote server" + "Settings…" button | This computer | Settings dialog: server address, token, "Test connection" |
| Run / Cancel | Button | | Run turns into Cancel while running |
| Progress | Progress bar + status line | | e.g. `Apple GPU · image 3/5 · SAM proposals` or `Remote (lab-gpu) · image 3/5 · SAM proposals` |

**Right panel: results**

| Element | Widget | Source |
|---|---|---|
| Image picker | Dropdown | Processed images |
| Overlay | Image view with zoom/pan; "Show particle IDs" checkbox | `measure/<panel_id>__overlay.png` / `__overlay_ids.png` |
| Histogram | Matplotlib canvas (`FigureCanvasQTAgg`) | Equivalent diameters from Step 6 (nm, or px if no scale) |
| Summary | Small table | n, median, mean, SD, D10, D90, unit, scale source |
| Warnings | Text label | §7 |
| Particles | Table | This image's rows from `particles_all.csv` |
| Buttons | "Open results folder", "Save as zip…" | |

The window must open fast: **do not import torch or SAM in the GUI thread at startup.** Import them inside the worker on first run.

---

## 3. Change to existing code

Add an optional `progress` callback to `run_cleanup_stage` in `run.py`:

```python
progress: Callable[[int, int, str, str], None] | None = None   # (image_index, n_images, panel_id, stage)
```

- Call it when each stage starts for each panel: `cleanup`, `proposals`, `graph`, `resolve`, `measure`.
- `None` means behavior is unchanged. The CLI doesn't pass it.
- The callback is also how **cancel** works: if it raises an exception, the run stops there and the exception propagates.

---

## 4. Local mode ("This computer")

When the user clicks Run:

1. Create a run folder `~/LNP-SAM runs/<YYYY-MM-DD_HHMMSS>/`.
2. Copy the images to `<run>/inputs/`. Write `<run>/manifest.csv`:
   `panel_id` = sanitized filename stem (add `_2`, `_3`… for duplicates), `paper_id` = `app`, `image_path`, `modality`, `nm_per_px` (blank if not given).
3. In a **`QThread` worker**, call
   `run_cleanup_stage(manifest, config, run_folder, run_step2=True, run_step3=True, run_step4=True, run_step6=True, device=<auto>, progress=<callback>)`.
   The callback emits a Qt signal to update the progress bar, and raises `RunCancelled` if Cancel was clicked.
4. Write `run_info.json` (§8) and show the results (§2).

- Config: the repo's `config.yaml`. Resolve `sam.checkpoint` relative to the repo root; `--checkpoint PATH` overrides it.
- One run at a time: input controls are disabled while a run is going; the Run
  button becomes Cancel. After a cancellation request it becomes disabled
  "Cancelling…" until the worker reaches the next cancellation boundary (§12.6).
- Cancelled runs show "Cancelled"; the partial folder is left on disk.

**Device**
- Use `select_device("auto")` and show the result in the status line.
- Before importing torch: `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")`.
- If the run fails on `mps`, show "Apple GPU failed, retrying on CPU" and rerun once with `device="cpu"`.
- `--device cpu|mps|cuda` launch flag to force a device.

---

## 5. Remote mode ("Remote server")

### 5.1 The server (`lnpsam/server.py`)

The same pipeline without a window, run on a machine with a GPU:

```bash
python -m lnpsam.server --host 0.0.0.0 --port 8765 --token <secret>
```

Built with FastAPI + uvicorn. Every request needs `Authorization: Bearer <token>`, otherwise `401`.

| Endpoint | Does |
|---|---|
| `GET /info` | Returns LNP-SAM version, git commit, `checkpoint_sha256`, device |
| `POST /jobs` | Multipart: image files + `modality` + `nm_per_px`. Saves them, queues the job, returns `{job_id}` |
| `GET /jobs/{id}` | Returns `state` (`queued`, `running`, `done`, `failed`, `cancelled`), queue position, and the latest progress `(image_index, n_images, panel_id, stage)` or the error message |
| `GET /jobs/{id}/result` | `results.zip`: `measure/`, `particles_all.csv`, `panels_all.csv`, `run_info.json` |
| `DELETE /jobs/{id}` | Cancels a queued or running job and deletes its files |

- The server runs a job exactly like local mode (§4 steps 1–4) in a background thread, using the same `progress` callback to record progress and to stop cancelled jobs.
- **One job at a time**; others wait in a queue.
- Max upload 200 MB.
- A job's files are deleted after its result is downloaded, or after 24 h.

### 5.2 The client (`lnpsam/remote.py` + Settings dialog)

- Settings dialog: server address (e.g. `http://lab-gpu:8765` or `http://localhost:8765` through an SSH tunnel), token, "Test connection" (calls `/info` and shows the server's device and version).
- On Run (in the worker thread): upload the images → poll `GET /jobs/{id}` every second and emit the same progress signal as local mode → download `results.zip` → unzip into a local run folder `~/LNP-SAM runs/<timestamp>/` → display it with **the same code as local mode**.
- Cancel calls `DELETE /jobs/{id}`.
- If the server's LNP-SAM version or `checkpoint_sha256` differs from the app's, show a warning ("Server runs a different LNP-SAM version; results may differ") with Continue / Cancel.
- The first time a user picks a remote server, tell them once: "Your images will be sent to <server>."
- Use `requests` for HTTP.

### 5.3 Where the server can run (README section)

| Setup | How |
|---|---|
| Lab GPU computer | Start the server there; others enter `http://<that-computer>:8765` + token |
| HPC | Start the server on a compute node (batch job), then on your own computer `ssh -L 8765:<node>:8765 <login-node>` and enter `http://localhost:8765`. Example Slurm script in the README |

Security note for the README: v1 has no HTTPS. Use it on a lab network or through an SSH tunnel, never exposed to the open internet.

---

## 6. Install and launch (README section)

```bash
conda create -n lnpsam python=3.11 -y
conda activate lnpsam
pip install -r requirements.txt     # adds PySide6, requests; server also needs fastapi, uvicorn, python-multipart
# put SAM ViT-B weights at weights/sam_vit_b_01ec64.pth (download link in README)
python -m lnpsam.gui                # the app
python -m lnpsam.server ...         # only on the remote machine
```

- Pin exact versions of the new packages in `requirements.txt`.
- The app opens its window before checking local prerequisites. In local mode,
  check the SAM weights and configured `checkpoint_sha256` in a background worker;
  show the message from §7 if the check fails. Repeat verification before loading
  SAM as required by the cache spec. Neither file hashing nor torch/SAM imports
  may block initial GUI rendering. The server checks its own weights before
  accepting jobs. In remote mode the app does not require local weights (§12.6).
- EasyOCR downloads its models the first time it runs (internet needed once, on whichever machine runs the pipeline).

---

## 7. Plain-language warnings

| Condition | Message |
|---|---|
| No scale found and none entered | "No scale bar found. Sizes are in pixels. Enter nm per pixel to get nm." |
| Too few particles for stats | "Too few particles for statistics on this image." |
| `FRAME_PARTICLE` flag | "A particle fills most of the image; check the result." |
| `POSSIBLE_CORE_ONLY` flag | "Some particles may be detected by their core only." |
| Image long side > 1024 px | "This image is large (W×H). SAM shrinks it to 1024 px, so small particles may be missed. For best results, crop to the area you care about." |
| SAM weights missing or wrong | "SAM model file not found / doesn't match. Download it from <link> to <path>." |
| Remote server unreachable / wrong token | "Can't reach the server" / "The server rejected the token." |

---

## 8. `run_info.json`

Written into each run folder: date, LNP-SAM git commit (if available), where it ran (`this computer` or the server address), device, full `config.yaml` contents, `checkpoint_sha256`, and per image: scale source, `nm_per_px`, particle count.

---

## 9. Tests

| Test | What it checks |
|---|---|
| Progress callback | Called for each stage of each panel; CLI output unchanged when `progress=None`; raising inside it stops the run |
| Server | With FastAPI's `TestClient` and SAM faked (same pattern as `tests/test_run.py`): submit a job, poll to `done`, download a zip with the expected files; `401` without the token; `DELETE` cancels |
| Remote client | Start the server on a random local port in a background thread; run the client end to end with SAM faked; the unzipped folder matches a local run on the same images |
| GUI smoke test | `pytest-qt`: window opens without importing torch; add an image, run with SAM faked, results panel fills in |
| Manual | 3 golden panels give the same `particles_all.csv` via the app (local), the app (remote), and the CLI |

---

## 10. Done when

- [ ] `python -m lnpsam.gui` opens the window; adding golden panels and clicking Run shows overlay, histogram, summary and table for each.
- [ ] Remote mode works against a server on another machine (or an HPC node through an SSH tunnel).
- [ ] Results folders from local and remote runs look the same and display the same.
- [ ] Works on Yash's Mac (Apple GPU or CPU) and on a CUDA machine.
- [ ] App results (local and remote) equal CLI results on the 3 golden panels.
- [ ] All tests pass; CLI behavior unchanged.
- [ ] The visual, interaction, accessibility and resource targets in §12.15
  have been checked on a Mac and a CUDA workstation; measurements are recorded.

---

## 11. Later, only if needed

Installers + code signing · HTTPS for the server · editing particles · tiling large images · DM3/DM4 files · sharing the SAM cache across runs.

---

## 12. GUI design and interaction specification

**Status:** design specification. This section does not claim that the resource
targets have been met; validation results are recorded in
`design/gui-validation.md`. It fills in the visual decisions left open in §2. The existing
pipeline remains the source of every scientific result.

### 12.1 Design intent

Build a quiet scientific desktop tool. The image and its measurements have the
most visual weight; the interface provides a stable frame around them. A user
should be able to add images, choose where to compute and run without learning
a separate navigation system.

- **Dark by default, and fixed dark in v1.** Near-black and charcoal surfaces,
  light neutral text, restrained Texas Tech scarlet accents.
- **Conventional controls.** Visible labels, ordinary radio buttons, a number
  field, dropdowns, a list, a table and a single primary action. No dashboard
  landing page, onboarding carousel or hidden icon-only navigation.
- **Simple hierarchy.** Prepare on the left; inspect on the right. Keep input
  order identical to the order in which the user makes decisions.
- **Minimal rendering.** Flat fills, a few thin dividers, small corner radii and
  no decorative animation. Render changes when data or user input changes.
- **Human wording.** Sentence case, brief specific messages and familiar units.
  Do not call the app an "AI assistant", invent confidence scores, show marketing
  copy or describe a scientific warning as a failed run.

Texas Tech references are limited to color and typography. Use the text name
**LNP-SAM** as the product identifier. A university logo, seal or recreated Double T
is not required for v1; no large banner competes with the microscope image.

### 12.2 Color system

The current [Texas Tech digital guidelines](https://www.ttu.edu/brand/websites/index.php)
specify scarlet **#E90802**, black **#000000** and white **#FFFFFF**. This app uses
those for its small identity accents and primary action. The charcoal surfaces
and semantic warning colors below are application choices, not additional
official TTU brand colors.

| Token | Hex | Use |
|---|---|---|
| `brand.scarlet` | `#E90802` | Run fill, checked indicator, short selected-tab underline, progress fill |
| `brand.black` | `#000000` | Compact product header |
| `brand.white` | `#FFFFFF` | Text on primary red action |
| `surface.window` | `#121214` | Main window and results workspace |
| `surface.panel` | `#1B1B1F` | Input rail, dialogs and analysis area |
| `surface.field` | `#222226` | Inputs and ordinary buttons |
| `surface.hover` | `#2A2A30` | Hovered ordinary controls and rows |
| `surface.pressed` | `#151517` | Pressed ordinary controls |
| `surface.selected` | `#37373F` | Selected file/table row; retains light text |
| `surface.viewer` | `#0B0B0D` | Letterboxing around the image |
| `border.divider` | `#44444B` | Noninteractive panel and row dividers |
| `border.control` | `#707078` | Essential input, button and checkbox boundaries |
| `text.primary` | `#EDEDED` | Labels, values, filenames and ordinary button text |
| `text.secondary` | `#B5B5BD` | Supporting labels, units and provenance |
| `text.hint` | `#93939F` | Placeholder/helper text; still readable |
| `text.disabled` | `#777780` | Inactive control text only |
| `action.hover` | `#C90702` | Hovered primary action |
| `action.pressed` | `#AA0602` | Pressed primary action |
| `focus` | `#E90802` | Keyboard-focus outline on dark controls |
| `focus.on_selected` | `#EDEDED` | Focus cue on selected rows or the red primary action |
| `warning.background` | `#30291C` | Inline scientific warning |
| `warning.foreground` | `#E9C46A` | Warning label/icon; body can remain light neutral |
| `error.background` | `#361B1D` | Inline error surface |
| `error.foreground` | `#FFB4B0` | Error label/icon on that surface |

Keep scarlet concentrated in the action and selection cues, approximately a few
percent of the interface area. Do not fill the input rail red, put every table
value in red, or make all particle outlines red. Ready and Complete states use
neutral text with explicit words; they do not need a permanent green badge.

The red Run button uses **white** text, not `text.primary`: calculated sRGB
contrast is about 4.67:1. Hint text on `surface.field` is about 5.22:1;
`border.control` against that surface is about 3.23:1. Scarlet on a charcoal field
is about 3.40:1, so it is a useful non-text accent but not a small essential text
color. Never encode warning, selection, completion or exclusion by color alone.

Use the [W3C text contrast](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html)
and [non-text contrast](https://www.w3.org/WAI/WCAG22/Understanding/non-text-contrast.html)
ratios as palette targets: at least 4.5:1 for ordinary text and at least 3:1 for
essential control boundaries and focus cues. This is not a claim of complete
WCAG conformance for the future desktop app. Recheck actual rendered states,
including disabled controls, tooltips, selections and native popup menus.

### 12.3 Typography and numbers

TTU's digital guide names Neue Plak as the primary digital family and **Inter for
UI components**, with Tahoma as an alternative. The
[visual-identity guide](https://www.ttu.edu/brand/visual-identity/)
also lists Helvetica Neue. For a dense scientific interface use one understated
sans-serif family throughout; no collegiate display lettering in form labels.

Choose the first locally installed family in this order:
**Inter → Helvetica Neue → Tahoma → Arial → Qt system GeneralFont**.
Resolve the family once with `QFontDatabase`, not while painting cells. Helvetica
Neue is the practical fallback on this Mac. Do not download fonts at startup,
require internet access for typography or bundle proprietary font files as part
of this design. If a licensed Neue Plak installation is later supplied, it can
be considered for the small product name; it is not a dependency.

All geometry below is in **logical pixels**, before device-pixel-ratio scaling.
Reference type sizes are visual targets, not hardcoded physical-pixel fonts.
Translate them to Qt font metrics and respect OS display/text scaling.

| Role | Reference size | Weight | Treatment |
|---|---|---|---|
| Product name | 18 px | Semibold | `LNP-SAM`; no oversized title or subtitle |
| Pane title | 15 px | Semibold | `Inputs`, `Results` |
| Group/tab/table heading | 13 px | Semibold | Sentence case |
| Buttons, fields, labels, table cells | 13 px | Regular; primary button semibold | Main interface text |
| Metadata and helper text | 12 px | Regular | Light secondary text |
| Chart tick labels | 12 px | Regular | Never smaller than 11 px |
| Short technical details | 12 px | System monospace, if needed | Paths and versions only, not all numbers |

Do not use extra-light weights, wide tracking, all-capital group labels or
multiple competing typefaces. Use native numeric alignment/tabular figures
where available. Scientific values are right-aligned; labels and units are
left-aligned. Show a value such as `153.91 nm` rather than a very large number
separated from a tiny unit.

### 12.4 Window geometry and resizing

| Part | Default/limit | Behavior |
|---|---|---|
| Window | 1280 × 840 | First launch; restore last geometry within current screen bounds |
| Minimum supported desktop size | 1024 × 720 | Native app target; smaller inline preview can reflow independently |
| Product toolbar | 44 high | Plain product name left; brief run/mode status; no custom OS window buttons |
| Main split | Horizontal `QSplitter` | Left input rail and right results workspace |
| Input rail | 296 wide; adjustable 264–360 | Keeps input labels and controls readable |
| Splitter handle | 6 wide | Native resize interaction; quiet divider |
| Pane padding | 16 | Consistent, not large empty margins |
| Group separation | 24 | Images, modality/scale, compute, Run/progress |
| Related-control gap | 8 | Label-to-control and sibling controls |
| Label-to-next-group gap | 16 | Where a full 24 is unnecessary |
| Field/ordinary button height | 32 minimum | Grow with actual font metrics/OS scaling |
| Run/Cancel height | 40 | Full input-rail content width; easy to locate |
| Control corner radius | 3 | No pill buttons or rounded cards |
| Analysis tabs | 32 high | `Distribution`, `Particles` |
| Results split | Vertical `QSplitter` | Viewer above, analysis area below |
| Initial viewer/analysis allocation | About 65% / 35% | Viewer remains dominant; user can resize |
| Viewer minimum height | 220 | Analysis area may scroll rather than crush the image |
| Analysis minimum height | 160 | Chart/table remain operable |
| Bottom status line | 28 minimum | One short status; can grow if text scaling needs it |

Use layouts and size policies instead of absolute coordinates. Set sensible
minimums, permit native scrolling for a long input list/table, and let helper
messages wrap. At 1024 × 720 retain the two main panes and both analysis tabs;
do not shrink fonts or hide Run to fit. Long image names are middle-elided,
with the full name/path available in a tooltip. A narrow distribution area
stacks its summary under the chart. Toolbars wrap or grow when labels no longer
fit at increased text scale.

The OS owns window decoration, fullscreen behavior and file dialogs. Do not
imitate macOS traffic lights or build a custom frameless window. Save window
geometry, splitter positions, last compute choice and table column widths in
`QSettings`; do not store particle data or tokens there.

### 12.5 Input rail: exact controls and wording

Keep the order **Images → Modality → Scale → Compute → Run → Progress**. The
first view contains no results and no connection dialog unless the user asks.

| Control | Design and interaction |
|---|---|
| Images heading | `Images` plus a quiet count such as `3 added` |
| Add images… | Ordinary outlined button, always visibly labeled; native multi-file picker |
| File list | Native model-backed list. Filename on the first line; `TEM · 768 × 512` only after modality is known, or just dimensions, on the second. No eagerly decoded thumbnails |
| Empty file list | `Add microscope images` and `PNG, JPG or TIF · drag files here`; no illustration, dashed animated dropzone or oversized upload icon |
| Drag-over | Brief static highlight of the list boundary; no full-window overlay |
| Remove | Small labeled `Remove` action for selected input files. Delete/Backspace works only while the file list has focus. Removing an input never deletes the original file |
| Large image | Small warning indicator beside dimensions and full §7 warning in a tooltip/inline message; color is accompanied by text |
| Modality | Ordinary TEM/SEM radio controls, TEM selected initially. Helper: `Applies to all images.` |
| Scale | Label `Scale (nm per pixel)`; blank numeric field with helper `Leave blank to read the scale bar.` Empty and zero are not equivalent: zero/negative/nonfinite entries are invalid |
| Scale scope | Helper `Applies to all images in this run.` Do not infer per-image manual scales from one field |
| Compute | Ordinary dropdown with exact options `This computer`, `Remote server` |
| Settings… | Ordinary button beside/below compute control, enabled for Remote server. Opens only the connection dialog in §12.10 |
| Mode helper | Local: `Uses an available GPU, otherwise CPU.` Remote: configured server name/address plus `Images are sent to this server.` |
| Run | Scarlet fill/white text. Disabled until inputs and selected compute-mode prerequisites are valid |

Validate the scale when the user leaves the field or attempts Run, not on every
partial keystroke. Show `Enter a positive number, or leave blank for automatic
scale detection.` below the field; focus the field after an attempted invalid
run. Use the decimal conventions of the user's locale, then serialize the
numeric value consistently. File selection/dimension reads occur without SAM.

Modality and scale are frozen for a running batch. Finished results always show
the settings recorded for that run; changing input controls prepares a future
run and cannot relabel an old result. Do not silently recalculate completed
measurements when the manual-scale field changes.

### 12.6 Run, progress and cancellation states

| State | Action/control state | Visible message and results behavior |
|---|---|---|
| Empty | Run disabled; input controls enabled | `Add images to begin.` Results area: `Results will appear here after a run.` |
| Checking local setup | Run disabled; compute remains selectable | `Checking local model…` in rail; check file/hash on a worker, never block opening the window |
| Local setup unavailable | Run disabled in local mode | Specific missing-model/hash error plus a usable path/link; user can still switch to Remote server |
| Ready | Run enabled; inputs enabled | `Ready · 3 images` or equivalent; no pulsing indicator |
| Remote queued | Main action Cancel; inputs frozen | `Queued · position 2` only when the server reports that position |
| Upload/download | Main action Cancel; inputs frozen | `Uploading images…` / `Downloading results…`; show byte progress only if measured |
| Running | Main action becomes ordinary outlined Cancel | `Apple GPU · image 3/5 · SAM proposals`; latest completed results can remain visible, clearly labeled |
| Cancelling | Main action `Cancelling…`, disabled | `Stopping after the current stage.` Do not promise immediate termination inside a SAM call |
| Cancelled | Run restored; inputs enabled | `Cancelled · partial results kept in <folder>` for local mode. In remote mode, describe cancellation/deletion according to the server response; never promise its files were kept |
| Failed | Run restored if prerequisites allow; inputs enabled | Specific error, Retry through Run, and a disclosure for diagnostic details; do not replace the whole app with a crash screen |
| Complete | Run restored; inputs enabled | `Complete · 5 images`; final results picker and export actions enabled |

The existing callback reports **stage starts**, not inner-stage progress. Name
stages `Cleanup`, `SAM proposals`, `Relations`, `Classification`, `Measurement`.
Use the progress bar for completed-image count, with the current image/stage
stated in text. At the start of image 3/5, the bar reflects two completed images;
at successful completion it reaches five. Do not assign arbitrary weights to
stages, show guessed percentages during SAM or invent an ETA. Resolve the final
completion count in the worker's completion signal.

No animated striped progress bar, spinner, heartbeat or continuously advancing
indeterminate progress animation. For unknown transfer/queue duration, a static
track and explicit text are sufficient. Coalesce repeated progress updates;
update immediately on meaningful stage/error/completion changes, no more than
five cosmetic updates per second. Remote polling may stay at the specified
one-second interval and stops after terminal state/cancellation.

When Cancel is clicked, request cancellation once and disable that action.
The window stays responsive until the worker acknowledges cancellation. Local
MPS fallback is explicit: `Apple GPU failed, retrying on CPU`; there is at most
one fallback attempt, and the device/status text changes. A cancelled job must
not be retried as an MPS failure.

### 12.7 Image viewer and image switching

Use a raster-backed `QGraphicsView`/single `QGraphicsPixmapItem` or equivalent
native raster image view. Do not add an OpenGL viewport. The image retains its
original proportions and pixel values; the dark theme applies to surrounding
chrome, not to scientific images. Existing overlay colors are preserved exactly.

Viewer toolbar, in order:

1. Label `Image` with processed-image dropdown, showing filename and index.
2. Checkbox `Show particle IDs`, initially unchecked.
3. `Fit` and `100%` ordinary compact buttons, followed by a quiet zoom readout.

Fit is the initial view. Center the image on `surface.viewer`, letterboxing
without stretch or crop. Wheel zoom anchors at the pointer; drag pans once the
image is zoomed. Fit restores centered full-image view; 100% means one source
image pixel per logical scene pixel with appropriate high-DPI handling. Keep
the source-image transform stable across base and ID PNGs. The ID image is
typically 3× resolution, but appended legend strips do **not** always have
exactly 3× the base legend height. Uniformly scale each raster by its known
overlay scale (or raster width divided by source-image width), and anchor its
image-content top-left to the same scene origin. Do not stretch the whole ID
raster to the base raster's width and height or fit each variant independently.
Use a stable scene extent consisting of the source image plus enough padded
legend space for either variant, or display legends separately. Toggling IDs
must not jump zoom, pan or the position/size of the microscope image.

Both view variants are display outputs, not masks to be regenerated in the GUI.
Load the ID PNG only when requested. Switching images updates the overlay,
statistics, warnings and table together; never leave the previous image's
statistics under the new image. Prefer atomic visible updates after the selected
panel is ready. Use a small `Loading image…` label for slow disk reads, not a
blocking modal or animated skeleton. Cancel/discard an obsolete load when the
user selects another image.

The dropdown contains only panels with completed results. When a later run is
processing, label the visible workspace `Previous run` or `Completed images in
this run` as appropriate, so it is never mistaken for the current stage's live
segmentation. No image editing, mask brushing, magnifier animation or simulated
live SAM outlines in v1.

### 12.8 Analysis area, plots and tables

Below the viewer, a plain `QTabWidget` provides **Distribution** and
**Particles**. Tabs are text labels with a short scarlet underline for the
active tab, not segmented pills. Distribution is selected when an image first
finishes. Both tabs share the selected image and its units.

**Distribution**

- One Matplotlib histogram and a compact label/value summary, beside each other
  when there is room, stacked when necessary. No row of oversized KPI cards.
- Plot equivalent diameters from **`in_stats == True`** particles using `d_nm`
  for a calibrated image or `d_px` otherwise. This matches the population used
  by Step 6's `stats_all`; aggregates do not become separate diameter observations.
- Use a fixed, deterministic bin rule shared across renders of the same data
  (e.g. NumPy `histogram_bin_edges(..., bins="auto")`, with a practical cap of 40
  bins). Replot only when the selected panel/data changes; resizing changes the
  canvas layout, not the underlying results or bin choices.
- Axis labels: `Equivalent diameter (nm)` or `Equivalent diameter (px)` and
  `Particles`. Counts are integers. A single neutral light-gray bar series is
  sufficient; scarlet can mark the median with a thin line and direct label.
  Never color bars by arbitrary categories or turn statistical quality into a
  colored score.
- Chart background follows `surface.panel`. Axis/tick text is light neutral;
  gridlines are thin and quiet. Remove top/right spines. No gradient bars, 3D
  charts, animated drawing, live toolbar by default or continuous mouse-hover
  recomputation. A toolbar is not needed to interpret this simple histogram.
- Summary shows `Detected`, `Used in statistics`, `Median`, `Mean`, `SD`, `D10`,
  `D90`, `Units`, and `Scale`. Detected is `n_particles`; used count comes from
  `stats_all.n` or `n_in_stats`. Keep these counts separate.
- Scale is explicit: `Automatic · 1.19 nm/px`, `Manual · … nm/px`, or
  `Not found · sizes in pixels`. Never display an uncalibrated result as nm.
- Read summary statistics from `panel.json`; do not invent a different GUI
  definition of sample SD/percentiles. Display to two decimal places for sizes,
  integer counts, and enough significant digits for nm/px. Raw saved values are
  unchanged. Null values are an em dash, never zero.
- If `stats_all` is null, show `Too few particles for statistics on this image.`
  and em dashes for the unavailable statistics. A histogram of available
  in-statistics diameters can still be shown with its sample count; do not invent
  the missing summary or silently ignore the warning.

**Particles**

- Read-only model-backed `QTableView`, not a widget per cell. Compact rows of at
  least 28 logical pixels, horizontal header, quiet horizontal dividers and no
  heavy grid. Column headers remain visible while scrolling.
- First visible columns: `ID`, `Diameter (nm/px as applicable)`, `Hollow`,
  `In statistics`, `Truncated`, `Touches exclusion`, `Aggregate ID`. Remaining
  CSV fields are available by horizontal scrolling; unit-bearing headers
  remain accurate for each image. Keep ID visible at the left.
- Display boolean values as `Yes`/`No`, not colored switches. Null values are
  an em dash; numeric sorting uses original numbers, not formatted strings.
- Clicking a header sorts that view only and does not rewrite saved outputs.
  A selected row uses neutral selection plus native focus. The existing
  Show particle IDs control lets the user match a row to the image; table
  selection does not require rebuilding the raster overlay or altering a mask.
- Right-align numbers; left-align text. Default size precision is two decimals.
  Exclusion/flags remain inspectable even for particles outside statistics.

`Open results folder` and `Save as zip…` are ordinary outlined buttons in a
stable bottom row of the results workspace. Keep them disabled before a
completed run exists; explain their purpose with a short tooltip. They act on
the displayed run, not on whichever image happened to be added last. Save uses
a native destination picker and background compression; do not block the GUI.

### 12.9 Control appearance and focus

| State/control | Appearance and behavior |
|---|---|
| Ordinary button | `surface.field`, light text, 1 px `border.control`, 3 px radius, horizontal padding 12 |
| Hover | Flat `surface.hover`; no enlargement, shadow, fade or movement |
| Pressed | Flat `surface.pressed`; keep text/boundary readable |
| Run | `brand.scarlet`/white; darker `action.hover`/`action.pressed`; 40 high |
| Cancel | Ordinary outlined button in the same location/size; cancellation is not styled as an alarming destructive red action |
| Disabled | Neutral fill and disabled text, no red fill; explanation remains readable outside the disabled control |
| Keyboard focus | Native focus behavior plus a clearly visible 2 px cue; scarlet on ordinary dark controls, `focus.on_selected` on selected list/table rows and the red primary button |
| Text/number input | Flat dark field with persistent visible label; units outside/within label, not replaced by a placeholder |
| Radio/checkbox | Small native indicator with checked mark plus readable label; scarlet active indicator is supported by shape |
| Dropdown/menu | Opaque dark popup, readable selected row and native keyboard navigation; do not style only the closed dropdown |
| List/table selection | Neutral `surface.selected`, light text, native focus indicator; no fully scarlet rows |
| Tooltip | Compact opaque dark native surface, readable text; supplements rather than replaces an essential label |
| Scrollbar | Native geometry/behavior with a visible thumb; it must remain usable in the fixed dark theme |

One primary action per context: Run in the main window, Save in the Settings
dialog, Save in a native export dialog. There are no floating action buttons,
large icon-only circles or decorative custom icons. Use standard Qt icons only
where they improve recognition; text buttons are the default. Never rasterize
labels, chart axes or table cells into UI background images.

### 12.10 Remote Settings and confirmation dialogs

The Settings dialog is a small native modal, approximately 480 wide, with a
minimum width of 400 and content-driven height. It uses the same font, spacing
and controls as the main window, not a second visual theme.

Order:

1. Title `Remote server`.
2. Persistent label `Server address` and text field; helper example
   `http://localhost:8765`.
3. Persistent label `Access token` and password-echo field. An ordinary
   `Show token` checkbox can reveal it briefly on user request.
4. Ordinary `Test connection` button and an inline result area.
5. Bottom action row `Cancel` and `Save` (primary), aligned right.

Connection results are concrete: `Connected · CUDA · LNP-SAM 0.1.0` or the
§7 unreachable/token message. During a test show `Testing connection…`, disable
only duplicate connection tests, and perform network work on a worker. Cancel
closes without applying edits. Save applies the address/token; saving is not
the same as submitting images. The address may be persisted; the token remains
in memory for the session and is not written to QSettings, `run_info.json`
or logs. Persistent secure token storage can be a later change.

Before the first upload to a chosen server, use the spec's single confirmation:
`Your images will be sent to <server>.` Actions `Continue` and `Cancel`, with
server identity plainly visible. A server version/model mismatch uses a native
warning dialog containing the specific differing version/model identity and
the same choices. Do not present unrelated technical configuration options or
add server-administration screens.

Version/model identity belongs in connection details, not the normal image
view. Avoid showing a 64-character hash in the primary work area. Remote mode
never opens a missing-local-model dialog or requires local torch/SAM.

### 12.11 Warnings, errors and empty results

Place per-image scientific warnings directly beneath the image toolbar or
immediately above analysis. Use one compact wrapping warning area with a
label/icon and the exact plain-language messages in §7. A warning persists
with its image until the user switches images; it is not a toast that vanishes.

Distinguish:

- **Scientific caution:** scale missing, too few particles, frame particle or
  possible-core-only detection. Results are still valid outputs to inspect.
- **Input/setup problem:** invalid scale, missing/wrong weights, placeholder
  checkpoint hash or remote connection issue. Show near the related input.
- **Processing failure:** name the image and stage, retain the workspace and
  provide a concise action. Expandable `Details` may expose the traceback for
  troubleshooting; do not put it in the normal status line.

If no particles were detected, say `No particles detected in this image.`
Keep the image visible; show an empty table with headers and no misleading
zero-size bars/statistics. If the scale is absent, all relevant headers/labels
switch to pixels together. Never replace missing values with made-up examples.

### 12.12 Keyboard and accessibility

- Follow native tab order through image actions, modality, scale, compute,
  Settings, Run, results picker, ID toggle, viewer actions, analysis and export.
  Labels have buddies/accessible names; tooltips are not the only name.
- Native radio/checkbox/dropdown keys work as expected. Enter/Space activates
  the focused control, not an unexpected global analysis run.
- `Ctrl/Cmd+O` adds images. `Ctrl/Cmd+Enter` invokes Run only when enabled.
  `Ctrl/Cmd+S` opens Save as zip only with completed results. `F` fits the image
  only while the viewer has focus. Escape closes a dialog first; with the main
  window active it requests cancellation only for an active run.
- Table/list selection, check states, queued/running/failed status and warning
  reasons are expressed with text/shape as well as color.
- Keep focus visible on every active control; do not remove native focus
  painting. Test increased OS text scale without clipping labels/actions.
- Describe the image view accessibly using image name, dimensions and particle
  count; the table and summary provide text equivalents to the visual output.
- Progress announcements are limited to stage/image/terminal changes. Do not
  flood assistive technology with every poll, hover or zoom increment.

### 12.13 Rendering and memory contract

The dark theme itself does not guarantee low resource use. The following
implementation choices are required to keep the interface economical:

1. **Native raster widgets.** QWidget/PySide6, standard Qt layouts and a small
   palette/QSS layer. No QtWebEngine, Electron, Qt Quick scene, WebGL/OpenGL
   viewport, shaders or GPU image-processing effects. The OS compositor may
   still use the GPU; do not claim zero GPU usage for a desktop window.
2. **No decorative effects.** No gradients, glass/blur, backdrop effects,
   QGraphicsDropShadowEffect, animated transitions, glow, looping spinners,
   live-background textures or continuous repaint timers.
3. **Lazy result loading.** Decode only the selected panel's base overlay and,
   when requested, its ID overlay. Do not decode all input thumbnails or all
   processed images at startup. Discard obsolete loads.
4. **Bounded cache.** Limit decoded display-image caches to 64 MiB in total,
   evict least recently used panels, and account for decoded RGBA storage rather
   than compressed file size. Very large ID images get a display-resolution
   preview if necessary; retain the original file for exports. The GUI never
   changes the scientific data to make a preview smaller.
5. **One chart canvas.** Create Matplotlib lazily when Distribution is first
   opened; reuse one FigureCanvas/Axes. `draw_idle()` follows a data/layout
   change, never an animation loop. Close/release stale figure resources.
6. **Model-backed data views.** One shared dataframe/model for the current
   results, no per-cell widgets or eager duplicate formatted-string tables.
   Read required columns lazily where practical; preserve raw numeric values.
7. **Worker I/O and computation.** Hashing, image decode into QImage/bytes,
   CSV parsing, upload/download and zip compression occur off the GUI thread.
   Create/use widgets and QPixmap on the GUI thread. Signals carry small status
   records or references to prepared display data, not repeated SAM tensors.
8. **Separate science from display.** Import torch/SAM only in the processing
   worker when local work actually requires it. Changing tabs, resizing the
   window or toggling IDs never runs SAM, classification or measurement.
9. **No retained model references in UI objects.** Release worker-held tensor/
   model references when the local run lifecycle permits; do not put them in
   table models, thumbnails, signals or global GUI caches.
10. **Quiet idle.** No app-owned periodic repaint or polling timers when no run/
    network task is active. Native caret, tooltip and accessibility behavior
    remains enabled. Resize/pointer events repaint only affected widgets.
    Coalesce remote/progress updates; stop polling once the job is terminal.

### 12.14 Qt implementation map

| Design part | Qt implementation | Notes |
|---|---|---|
| Window/chrome | `QMainWindow`, native title bar, compact toolbar/statusbar | Standard OS behavior |
| Layout | Horizontal/vertical `QSplitter`, `QVBoxLayout`/`QFormLayout` | Persist ratios, avoid fixed positioning |
| Theme | Dark `QPalette`, small shared QSS/token module | Standard widget style; no custom rendering for every control |
| Input list | `QListView` + model/delegate | Text/dimensions only; native scrolling |
| Mode/scale | `QRadioButton`, `QDoubleSpinBox` or validated numeric field, `QComboBox` | A truly blank optional scale must be representable |
| Actions | `QPushButton` | Native hover/focus/disabled behavior with shared styling |
| Image | Raster `QGraphicsView` + one `QGraphicsPixmapItem` | Shared source transform and stable padded scene extent for both overlay variants |
| Analysis | `QTabWidget`, one Matplotlib `FigureCanvasQTAgg` | Lazy canvas construction |
| Summary | Small read-only label/value grid or table | No card widget per statistic |
| Particles | `QTableView` + `QAbstractTableModel` | Numeric sorting, right alignment |
| Connection | `QDialog`, labeled fields and inline status | Worker performs test; no hidden uploads |
| Worker signals | `QThread`/Qt signal handoff | GUI thread owns GUI objects |
| Preferences | `QSettings` | Geometry/view preferences only; no access tokens |

`design/lnpsam-ui-tokens.json` records palette/geometry/type targets for the
implementer; it is not a new runtime requirement. A future small `theme.py`
can translate these values into QPalette, QFont and QSS without introducing
third-party styling packages.

### 12.15 Review and acceptance checklist

These are **targets to measure on the working GUI**. Use the same machine, same
inputs and the same baseline pipeline configuration for comparisons.

| Check | Acceptance target |
|---|---|
| Initial appearance | Window appears within 2 seconds on the target Mac/workstation, excluding first-run environment installation; local hashing proceeds after it appears |
| Startup import boundary | GUI can open with torch/SAM imports forbidden and no local checkpoint; Remote mode still usable |
| Idle CPU | No app-owned periodic repaint/poll timers; aim below 1% of one CPU core over a settled 60-second idle observation; record actual process usage |
| UI memory | Record GUI-only resident memory before loading science libraries; aim at or below 200 MiB incremental RSS, excluding OS shared/compositor accounting; document the measurement method |
| Display cache | Decoded display-image cache stays at or below 64 MiB; rapid image switching does not grow memory without bound |
| During SAM | UI remains interactive; progress handling is small; no background chart redraws. Compare analysis wall time against CLI and investigate material regressions, with a target under 5% GUI overhead on the same warm-run workload |
| Chart/table updates | Selecting a panel, toggling IDs and sorting a table do not call pipeline stages or alter result files |
| High DPI/text scale | Check normal and 2× display scale plus increased text size; labels, focus rings, menus and dialog actions do not clip |
| Contrast/states | Verify palette ratios and readable focus, placeholder, hover, pressed, warning and selected-row states |
| Empty/error cases | Empty input, zero detections, no scale, too few particles, invalid scale, missing hash/model, unreachable server and wrong token have usable layouts |
| Run/cancel | No duplicate runs; Cancel remains reachable; Cancelling state is honest; a cancelled run is not retried as an MPS failure |
| Numeric consistency | Histogram in_stats population and summary equal Step 6; units are explicit; null summaries stay missing |
| Inspection | Base/ID image toggle preserves the scene transform; filenames/IDs remain identifiable; exported scientific results are untouched |

Review the default completed screen, the empty screen, an active run, the
Cancelling state, a missing-scale result and the Remote Settings dialog on each
target machine.
