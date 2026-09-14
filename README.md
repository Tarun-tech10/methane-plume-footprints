# Confirmed Methane Plume Footprints

From-scratch U-Net that decides whether a flagged methane enhancement survived
review and, if so, outlines it. Scores **0.68** on the held-out split, against a
reference convolutional segmentation network at 0.5761 and 0.0 for answering
"no plume" everywhere.

| file | what it is |
|---|---|
| `solution.py` | the entire pipeline, one self-contained file |
| `description.md` | method, measurements, and what was tried and failed |
| `submission.csv` | output of `solution.py` on the supplied test split |
| `check_submission.py` | validates a submission against the format rules |
| `requirements.txt` | numpy / pandas / pillow (torch installed separately, see below) |
| `DATASET_LICENSE.txt`, `DATASET_ATTRIBUTION.txt` | CC0 1.0, Harvard Dataverse provenance |

Run it with `python solution.py CMP_DATA submission.csv`. Details below.

---

# Running this on another laptop

`solution.py` is one self-contained file. It imports nothing from this project — only
numpy, pandas, torch and Pillow — so it runs anywhere with an NVIDIA GPU.

## 1. What the machine needs

| | requirement |
|---|---|
| GPU | **NVIDIA, CUDA-capable, >= 4 GB VRAM** (the run peaks near 2.8 GB) |
| RAM | >= 8 GB (it holds all 1827 scenes in memory, ~3.5 GB) |
| Disk | ~2.5 GB free (1 GB zip + 1 GB extracted) |
| Python | 3.9 or newer |

**The GPU is not optional.** The script hard-codes `DEV = 'cuda'` — that was required to
pass the platform's deterministic-execution check, which rejects a device probe. On a
machine with no NVIDIA GPU it will fail immediately at the first `.to(DEV)` (see
*Troubleshooting* if that is all you have).

Apple Silicon / AMD / Intel integrated graphics will **not** work.

## 2. Get the dataset

The code is in this repo; the **dataset is not** (1.0 GB, too large for a git repo).
Download `CMP.zip` from the [Releases tab](../../releases) and put it next to
`solution.py`.

If the Releases tab is empty the dataset has not been published yet — ask for
`CMP.zip` directly. Nothing here runs without it.

## 3. Set up, once

Check the GPU is visible:

```bat
nvidia-smi
```

If that prints a table with a GPU name, you are fine. Then:

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Now install torch — **the right build depends on how new the GPU is**:

```bat
:: RTX 50-series (5070, 5080, 5090 - Blackwell, sm_120).  MUST be cu128 or newer.
pip install torch --index-url https://download.pytorch.org/whl/cu128

:: RTX 20/30/40-series
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

On Linux/macOS the activate line is `source venv/bin/activate`.

### Verify properly — `is_available()` alone is not enough

On a 50-series card a too-old torch build reports `True` here and then **fails later with
"no kernel image is available for execution on the device"**, potentially an hour into the
run. So actually execute a kernel:

```bat
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability()); x=torch.randn(64,64,device='cuda'); print('kernels OK', float((x@x).sum()))"
```

This must print `kernels OK` followed by a number. On an RTX 5070 the capability is
`(12, 0)`. If it raises instead, the build has no kernels for this GPU — reinstall from
the cu128 index.

## 4. Unpack the data

```bat
python -c "import zipfile; zipfile.ZipFile('CMP.zip').extractall('CMP_DATA')"
```

You should end up with `CMP_DATA\train.csv`, `CMP_DATA\test.csv` and
`CMP_DATA\scenes\` holding 1827 PNGs.

## 5. Run it

```bat
python solution.py CMP_DATA submission.csv
```

That is the whole thing: it trains 5 models from scratch, calibrates on held-out data,
and writes `submission.csv`.

**Leave the machine alone while it runs.** Nothing else should use the GPU — sharing it
roughly halves the speed and can get the run killed.

### How long

Scales with the card. On the laptop RTX 3050 here the full run took **96 minutes**.

| GPU | rough estimate |
|---|---|
| RTX 3050 laptop | ~95-100 min |
| RTX 3060 / 4060 | ~50-65 min |
| RTX 4070 / 4080 | ~30-45 min |
| **RTX 5070** | **~25-40 min** |
| RTX 4090 / 5090 | ~20-30 min |

### What healthy output looks like

```
device cuda
train 1447  test 380
scenes read 1827
targets built  positives 0.565
fold 1 of 5 done            <- roughly every 1/5 of the total time
...
mask threshold 0.55  Dice 0.6272
accept fraction 0.590  held-out challenge score 0.7101
wrote submission.csv  rows 380  non-empty 224
```

The `held-out challenge score` near the end is the sanity check. **If it prints
something far below ~0.70, something went wrong** — do not submit that file.

## 6. Check the result before submitting

```bat
python check_submission.py CMP_DATA submission.csv
```

Must print `VALID` and `rows 380`. Then copy `submission.csv` back.

## Troubleshooting

**`torch.cuda.is_available()` is False** — you installed the CPU build of torch. Run
`pip uninstall torch` and reinstall with the `--index-url` line above.

**`CUDA error: no kernel image is available for execution on the device`** — this is *the*
RTX 50-series failure: the torch build predates Blackwell (sm_120). Fix with
`pip uninstall torch` then reinstall from the **cu128** index. Nothing else changes.

**`CUDA out of memory`** — the card is smaller than 4 GB or something else is using it.
Close other GPU programs. If it still fails, open `solution.py` and change `BATCH = 8` to
`BATCH = 4`; it will be slower and the result will differ slightly from ours, but it works.

**No NVIDIA GPU at all** — change `DEV = 'cuda'` to `DEV = 'cpu'` near the top. Be warned
this is roughly 15-20x slower (expect well over a day), so it is only worth it to prove the
script runs, not to produce a real submission. Do **not** submit a `solution.py` edited this
way: the device probe is exactly what the deterministic-execution check rejects.

**Run got killed with no error message** — something else killed the process, usually
another job claiming the GPU. Check with `nvidia-smi` and rerun on an idle machine.

## Note on reproducibility

Every seed is fixed, so rerunning on the *same* machine gives a byte-identical
`submission.csv`. A *different* GPU will give slightly different numbers — different cards
pick different kernels — but the score will be equivalent. That is expected and fine.
