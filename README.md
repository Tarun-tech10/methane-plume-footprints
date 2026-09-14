# Confirmed Methane Plume Footprints

## What this is, in plain words

Aircraft fly over the ground and measure methane in the air below. When they spot a
suspicious patch of gas, a human reviewer looks at it and decides: **is this a real gas
leak, or a false alarm?** If it is real, they draw around it.

This program does that job automatically. You give it the pictures, it tells you which
ones show a real leak and draws the outline.

It gets **0.68** out of 1.0. For comparison, the benchmark everyone is measured against
gets 0.5761, and answering "no leak" every single time gets 0.0.

You do **not** need to understand any of this to run it. Just follow the steps.

---

## What you need before starting

| | |
|---|---|
| A Windows PC | with an **NVIDIA graphics card** (GeForce RTX or GTX) |
| Graphics card memory | 4 GB or more |
| Computer memory (RAM) | 8 GB or more |
| Free disk space | about 3 GB |
| Time | 30 minutes to 2 hours, depending on the graphics card |

**The NVIDIA graphics card is not optional.** This will not work on a Mac, or on a
laptop with only Intel/AMD built-in graphics. To check what you have: press
`Ctrl+Shift+Esc` → **Performance** tab → look at the bottom left for **GPU**. It should
say NVIDIA something.

---

## Step 1 — Install Python (skip if you already have it)

Download from [python.org/downloads](https://www.python.org/downloads/) and run the
installer.

⚠️ **On the first screen, tick the box that says "Add Python to PATH"** before clicking
Install. If you miss it, nothing below will work and you will have to reinstall.

---

## Step 2 — Download the two pieces

**A. This code.** Green **`< > Code`** button at the top of this page →
**Download ZIP**. Right-click the downloaded file → **Extract All**.

**B. The data.** Go to the **[Releases](../../releases)** page on the right-hand side of
this repo and download **`CMP.zip`** (it is about 1 GB, so give it a few minutes).

Now **put `CMP.zip` inside the folder you just extracted**, next to `solution.py`. The
folder should look like this:

```
methane-plume-footprints/
    solution.py
    check_submission.py
    requirements.txt
    README.md
    CMP.zip          <-- the file you just downloaded
```

> If the Releases page is empty, the data has not been uploaded yet — ask whoever sent
> you this link for `CMP.zip`. Nothing works without it.

---

## Step 3 — Open a terminal in that folder

Open the folder in File Explorer. Click the address bar at the top (where the folder path
is), type `cmd`, and press **Enter**.

A black window opens. Everything from here is copy-paste into that window, pressing
**Enter** after each line, and **waiting for each one to finish** before the next.

---

## Step 4 — Set up (once only)

Copy-paste these two lines:

```bat
python -m venv venv
venv\Scripts\activate
```

You should now see `(venv)` at the start of the line. Then:

```bat
pip install -r requirements.txt
```

Wait for it to finish (a minute or so).

### Now install the maths library — pick ONE line

Which line depends on how new your graphics card is.

**If you have an RTX 5060 / 5070 / 5080 / 5090** (the newest generation):

```bat
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**For anything older** (RTX 4060, 3050, 3080, GTX, etc.):

```bat
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

This one is a big download (about 2.5 GB). Be patient.

---

## Step 5 — Check the graphics card actually works

This takes 5 seconds and saves you from discovering a problem an hour later. Paste it as
one single line:

```bat
python -c "import torch; print(torch.cuda.get_device_name(0)); x=torch.randn(64,64,device='cuda'); print('kernels OK', float((x@x).sum()))"
```

✅ **Good** — you see your graphics card name and the words `kernels OK`.

❌ **Bad** — you see `no kernel image is available`. You picked the wrong line in Step 4.
Fix it with:

```bat
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

---

## Step 6 — Unpack the data

```bat
python -c "import zipfile; zipfile.ZipFile('CMP.zip').extractall('CMP_DATA')"
```

Takes under a minute. You now have a `CMP_DATA` folder with 1827 images in it.

---

## Step 7 — Run it

```bat
python solution.py CMP_DATA submission.csv
```

**Now leave the computer alone.** Do not play games, do not run anything else heavy —
the program needs the whole graphics card. Using it for other things roughly halves the
speed.

### How long it takes

| Graphics card | roughly |
|---|---|
| RTX 3050 laptop | 95–100 minutes |
| RTX 3060 / 4060 | 50–65 minutes |
| RTX 4070 / 4080 | 30–45 minutes |
| RTX 5070 | 25–40 minutes |
| RTX 4090 / 5090 | 20–30 minutes |

### What you should see

```
device cuda
train 1447  test 380
scenes read 1827
targets built  positives 0.565
fold 1 of 5 done          <-- appears 5 times, evenly spaced
fold 2 of 5 done
...
mask threshold 0.55  Dice 0.6272
accept fraction 0.590  held-out challenge score 0.7101
wrote submission.csv  rows 380  non-empty 224
```

The `fold N of 5 done` lines are your progress bar. When you see `wrote submission.csv`,
it is finished.

⚠️ **Look at the `held-out challenge score` number.** It should be about **0.71**. If it
is much lower, something went wrong — do not use that file.

---

## Step 8 — Check the answer is valid

```bat
python check_submission.py CMP_DATA submission.csv
```

It must print **`VALID`** and **`rows 380`**. If it does, you are done — `submission.csv`
is your answer file. Send it back.

---

## If something goes wrong

**`'python' is not recognized`**
Python is not installed, or you missed the "Add Python to PATH" tick box in Step 1.
Reinstall it and make sure that box is ticked.

**`torch.cuda.is_available()` is False, or `Torch not compiled with CUDA enabled`**
You got the plain version of torch instead of the graphics-card version. Run
`pip uninstall torch -y` and redo the install line in Step 4.

**`no kernel image is available for execution on the device`**
Your graphics card is newer than the software you installed. This is the standard
RTX 50-series problem. Redo Step 4 using the **cu128** line.

**`CUDA out of memory`**
Something else is using the graphics card — close games, browsers with videos, other
Python windows. If it still happens, open `solution.py` in Notepad, find the line
`BATCH = 8` near the top, change it to `BATCH = 4`, and save. Slower, but it will run.

**It stopped with no error message**
Something else killed it, usually another program grabbing the graphics card. Just start
Step 7 again.

**No NVIDIA graphics card at all**
There is no practical fix. On a normal processor this would take well over a day. Borrow
a machine with a gaming graphics card.

---

## Notes for the curious

`description.md` in this folder explains how it works, what was measured, and — just as
importantly — the things that were tried and did **not** work.

Running it twice on the **same** computer gives a byte-for-byte identical answer; every
random choice is fixed in advance. A **different** computer gives slightly different
numbers, because different graphics cards do the arithmetic in a different order. That is
normal and the score comes out the same.

The data is public domain (CC0 1.0), from Harvard Dataverse —
see `DATASET_ATTRIBUTION.txt`.
