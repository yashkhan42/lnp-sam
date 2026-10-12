# LNP-SAM

Lignin nanoparticle measurements from TEM/SEM microscope images. The native
PySide6 desktop app runs the existing cleanup → SAM proposals → graph →
classification → measurement pipeline locally or through a remote server.

The GUI follows [the app specification](LNP-SAM_APP_SPEC.md) and its dark,
two-panel design. It includes image input, progress and cancellation, a zoom/pan
overlay viewer with particle IDs, a distribution plot, summary statistics, a
sortable particle table and zip export. Scientific measurements come from the
pipeline's saved outputs.

## Install and launch

Use Python 3.11, from the repository directory:

```bash
conda create -n lnpsam python=3.11 -y
conda activate lnpsam
pip install -r requirements.txt
python -m lnpsam.gui
```

With this checkout's virtual environment:

```bash
.venv/bin/python -m lnpsam.gui
```

The window opens before checking the local model. Torch, SAM and EasyOCR are
loaded only when processing begins. A remote-only client can use the GUI without
local weights. Remote server settings store the address; the access token stays
in memory for the session.

For local processing, download the
[official SAM ViT-B checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
and set both fields in `config.yaml`:

```yaml
sam:
  checkpoint: weights/sam_vit_b_01ec64.pth
  checkpoint_sha256: <the full SHA-256 of the model file>
```

Compute the hash with `shasum -a 256 weights/sam_vit_b_01ec64.pth` on macOS,
or `sha256sum weights/sam_vit_b_01ec64.pth` on Linux. The app refuses a missing,
placeholder or mismatched hash. Verification runs in a worker and repeats before
the pipeline loads the model. Checkpoint paths in the default config resolve
from the repository root; paths in a custom config resolve from its directory.
`--checkpoint` changes the path, while the hash remains in the config.

```bash
python -m lnpsam.gui --device cpu
python -m lnpsam.gui --checkpoint /path/to/sam_vit_b_01ec64.pth
python -m lnpsam.gui --config /path/to/config.yaml
python -m lnpsam.gui --results runs/mvp_v2
```

`--results` opens an existing results folder for inspection without running SAM.
Local runs are saved under `~/LNP-SAM runs/<timestamp>/`, with copied inputs,
manifest, config, pipeline artifacts and `run_info.json`. Cancel stops at the next
stage boundary and keeps the partial local folder. Apple GPU processing gets one
CPU retry after failure; a cancelled run is never retried.

EasyOCR downloads its models on its first processing run; internet access is
needed once on the computer executing the pipeline. For CUDA machines, install
the appropriate [PyTorch wheels](https://pytorch.org/get-started/locally/) for the
machine before installing the other requirements.

## Remote processing

On a lab GPU computer, install the requirements and configure its own weights and
hash, then launch:

```bash
python -m lnpsam.server --host 0.0.0.0 --port 8765 --token '<secret>'
```

The server verifies its weights at startup. In the desktop app, choose **Remote
server**, open **Settings…**, enter `http://<lab-computer>:8765` and the token,
then use **Test connection**. Before uploading, the app confirms the destination
and warns when known model/version identities differ. The server executes one
job at a time; additional jobs queue. Uploads are limited to 200 MiB.

Remote results download to the same local runs directory and use the same viewer.
The server deletes a job's files after download, cancellation, or 24 hours.
Cancellation during a pipeline stage takes effect at the next stage boundary.

For an HPC compute node, a Slurm script can start the server:

```bash
#!/bin/bash
#SBATCH --job-name=lnpsam-server
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# Adapt environment/project paths and Slurm account/partition to your cluster.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lnpsam
cd /path/to/lnpsam
# Export LNP_SAM_TOKEN when submitting the job; avoid putting it in a script.
python -m lnpsam.server --host 0.0.0.0 --port 8765 --token "$LNP_SAM_TOKEN"
```

On your own computer, open a tunnel to the allocated node:

```bash
ssh -N -L 8765:<compute-node>:8765 <login-node>
```

Use `http://localhost:8765` in app Settings. v1 does not configure HTTPS: use the
server on a lab network or through an SSH tunnel, never on the open internet.

## Tests and implementation status

```bash
pip install -r requirements-dev.txt
QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/lnpsam-mpl python -m pytest -q
```

Tests cover native GUI startup without torch/SAM, real pipeline stages with SAM
and OCR faked, progress/cancellation, result units/statistics, stable overlay
transforms, decoded cache bounds, settings lifecycle, and authenticated remote
jobs including a live loopback server. The loopback test needs permission to
bind a local port.

Real SAM equivalence on the three golden panels, CUDA hardware behavior,
increased OS text scaling and the full resource targets in §12.15 still require
manual measurement on the target machines. This implementation does not mark
the app specification's acceptance checklist complete.

Recorded layout checks, initial offscreen resource observations and remaining
acceptance work are in [GUI validation](design/gui-validation.md).
