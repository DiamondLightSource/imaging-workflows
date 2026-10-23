"""Build notebooks/EELS_insitu_processing.ipynb, the workflow notebook for Gatan multiframe EELS.

    python build_eels_insitu_nb.py ../notebooks/EELS_insitu_processing.ipynb
"""
import sys
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
md = lambda s: cells.append(new_markdown_cell(s.strip("\n")))
code = lambda s, tags=None: cells.append(
    new_code_cell(s.strip("\n"), metadata={"tags": tags} if tags else {})
)

md(r"""
# Gatan multiframe (in-situ) EELS: calibrate, align and save

## Usage

This notebook takes one Gatan multiframe STEM/EELS acquisition -- the folder of `.dm4` + `.raw`
files GMS writes for a time-series spectrum image -- and:

1. loads every stream (EELS high loss, low loss, HAADF, survey) **lazily and fully calibrated**
   in time, space and energy, with the `gatan_insitu` package;
2. optionally **aligns the frames** for specimen drift, measured on the HAADF and applied to
   every stream, so HAADF, high loss and low loss stay registered;
3. optionally **aligns the low loss on its zero-loss peak** and applies the same per-pixel
   energy shifts to the high loss;
4. saves each calibrated stream as an `.hspy` file;
5. writes an **analysis notebook** next to them, with code to lazily load the saved signals
   and explore them.

It is designed to be run non-interactively with papermill by the workflows server; every input
is set in the cell tagged `parameters` below and can be overridden with `-p name value`.

### Parameters

 visit: str

The path to the visit folder (e.g. `/dls/e02/data/2026/mg12345-1`)

 sample: str

The name of the sample folder

 dataset_name: str

The name of the folder holding the EELS multiframe data (e.g. `InSitu (5)`). It is searched for
as `<visit>/.../<sample>/<dataset_name>`, up to three folders deep, skipping `processing`

 align_zlp: bool

Align the low loss on its zero-loss peak and apply the same shifts to the high loss
(reads both streams twice: roughly 15-25 min for a 50-frame dual-EELS pair)

 align_frames: bool

Align the frames for specimen drift, measured on the HAADF, applied to all streams

**Optional**

 raw_data_path: str -- full path to the dataset folder; overrides the search

 output_subfolder: str -- outputs go to `<visit>/processing/[<output_subfolder>/]<path of the dataset under the visit>`

 frame_reference: str -- `'previous'` (default: each frame against the one before, robust to a changing specimen) or `'first'`

 drift_image: str -- name of the image stream to measure the drift on; blank = the HAADF series

 frames: int -- process only the first N frames (for a quick test); 0 = all

 compression: str -- `'gzip'` (default; about half the size for full-range EELS, more for sparse images) or `'none'` (faster)

 make_quicklook: bool -- re-read the saved files to draw an overview figure (checks they are good)

 gatan_insitu_path: str -- folder that *contains* the `gatan_insitu` package, if it is not installed

### Outputs

In `<visit>/processing/[<output_subfolder>/]<same folders as the raw data>/`:

- `<stream>.hspy` for every stream (e.g. `EELS_HL.hspy`, `EELS_LL.hspy`, `HAADF_Image.hspy`),
  calibrated, with the processing steps recorded in `metadata.Processing`
- `<dataset>_analysis.ipynb` -- a notebook that lazily loads these files and explores them
- `processing_summary.json` -- parameters, paths and alignment results
- `diagnostics/` -- figures (frame drift, ZLP before/after, quick look) and the measured shifts

### Dependencies

- gatan_insitu
- hyperspy, exspy
- numpy, scipy, scikit-image, dask
- matplotlib, pandas, nbformat
""")

code(r"""
# Parameters (overridden by papermill)
visit = ''
sample = ''
dataset_name = ''
align_zlp = False
align_frames = False

# optional
raw_data_path = ''
output_subfolder = ''
frame_reference = 'previous'
drift_image = ''
frames = 0
compression = 'gzip'
make_quicklook = True
gatan_insitu_path = ''
""", tags=["parameters"])

md("### Imports and logging")

code(r"""
%matplotlib inline
import os
import io
import re
import sys
import json
import glob
import time
import logging
import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import nbformat
import hyperspy.api as hs

if str(gatan_insitu_path).strip():
    sys.path.insert(0, str(gatan_insitu_path).strip())
import gatan_insitu as gi


def console_handler(stream='stdout'):
    # Log to the original console of the kernel, so messages also reach the workflow pod log.
    try:
        fh = getattr(sys, stream)._original_stdstream_copy
        return logging.StreamHandler(io.TextIOWrapper(io.FileIO(fh, 'w')))
    except (AttributeError, TypeError, OSError):
        return logging.StreamHandler(getattr(sys, '__' + stream + '__'))


logger = logging.getLogger('eels_insitu')
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    logger.addHandler(console_handler())

logger.debug(f'hyperspy {hs.__version__}, gatan_insitu {gi.__version__} from {gi.__file__}')
""")

md("### Tidy up the parameters and find the dataset\n\npapermill infers types from the command line (`0` is an int, `InSitu (5)` a string, `True` a bool), so coerce everything explicitly.")

code(r"""
def to_bool(x):
    if isinstance(x, str):
        return x.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    return bool(x)


def to_str(x):
    return '' if x is None else str(x).strip()


visit, sample, dataset_name = to_str(visit), to_str(sample), to_str(dataset_name)
raw_data_path, output_subfolder = to_str(raw_data_path), to_str(output_subfolder).strip('/\\')
align_zlp, align_frames = to_bool(align_zlp), to_bool(align_frames)
make_quicklook = to_bool(make_quicklook)
frame_reference = to_str(frame_reference).lower() or 'previous'
drift_image = to_str(drift_image)
frames = int(float(frames or 0))
compression = to_str(compression).lower()
compression = None if compression in ('', 'none', 'no', 'false', '0') else compression

if not visit:
    raise ValueError('visit must be given')
visit_path = Path(visit)


def find_dataset():
    # <visit>/[.../]<sample>/<dataset_name>, up to three folders deep, never inside processing
    if not (sample and dataset_name):
        raise ValueError('Give sample and dataset_name, or raw_data_path')
    tail = os.path.join(glob.escape(sample), glob.escape(dataset_name))
    found = []
    for depth in range(4):
        pattern = os.path.join(glob.escape(str(visit_path)), *(['*'] * depth), tail)
        for hit in glob.glob(pattern):
            rel = Path(hit).relative_to(visit_path)
            if rel.parts and rel.parts[0].lower() == 'processing':
                continue
            if Path(hit).is_dir() and any(Path(hit).glob('*.dm4')):
                found.append(Path(hit))
    if not found:
        raise FileNotFoundError(f'No folder {sample}/{dataset_name} with .dm4 files under {visit_path}')
    if len(found) > 1:
        raise ValueError(f'{len(found)} candidate folders, give raw_data_path to choose one: {found}')
    return found[0]


dataset_dir = Path(raw_data_path) if raw_data_path else find_dataset()
if not dataset_dir.is_dir():
    raise FileNotFoundError(f'{dataset_dir} is not a folder')
dataset_name = dataset_name or dataset_dir.name

# mirror the raw folder structure under processing
try:
    # absolute, but not resolve(): that follows symlinks (e.g. a linked visit) out of the visit
    rel_path = Path(os.path.abspath(dataset_dir)).relative_to(os.path.abspath(visit_path))
except ValueError:
    rel_path = Path(sample or 'sample') / dataset_name
    logger.warning(f'{dataset_dir} is not inside the visit; outputs go under {rel_path}')
out_dir = visit_path / 'processing' / output_subfolder / rel_path if output_subfolder \
    else visit_path / 'processing' / rel_path
diag_dir = out_dir / 'diagnostics'
diag_dir.mkdir(parents=True, exist_ok=True)

for name in ('dataset_dir', 'rel_path', 'out_dir', 'align_frames', 'align_zlp', 'frames'):
    logger.debug(f'{name}: {globals()[name]}')


def savefig(name, fig=None):
    (fig or plt.gcf()).savefig(diag_dir / name, dpi=150, bbox_inches='tight')


timings = {}
""")

md("### What is in the folder\n\nReads only the `.dm4` headers.")

code(r"""
print(gi.describe_folder(dataset_dir))
try:
    display(gi.summary_table(dataset_dir))
except ImportError:  # pandas missing: the text description above is enough
    pass
""")

md("### Load every stream, lazily and calibrated")

code(r"""
t0 = time.perf_counter()
signals = gi.load_insitu_dataset(dataset_dir, frames=frames if frames > 0 else None)
timings['load_s'] = time.perf_counter() - t0
if not signals:
    raise RuntimeError(f'Nothing could be loaded from {dataset_dir}')


def is_eels(s):
    return s.metadata.get_item('Signal.signal_type', '') == 'EELS'


def is_series(s):
    return 'Time' in [ax.name for ax in s.axes_manager._axes]


eels_keys = [k for k, s in signals.items() if is_eels(s)]
hl_key = next((k for k in eels_keys if 'HL' in k), eels_keys[0] if len(eels_keys) == 1 else None)
ll_key = next((k for k in eels_keys if 'LL' in k), None)
image_series = [k for k, s in signals.items() if not is_eels(s) and is_series(s)
                and s.axes_manager.signal_dimension == 2]

for key, s in signals.items():
    role = {hl_key: 'high loss', ll_key: 'low loss'}.get(key, 'image series' if key in image_series else 'still')
    print(f'{key:<28} {role:<12} {s}')
    logger.debug(f'loaded {key}: {s}')

if hl_key is None:
    logger.warning('No high-loss EELS stream identified')
""")

code(r"""
# Calibration of the main EELS stream: time from the recorded frame timestamps, space and energy from the tags.
main = signals[hl_key or eels_keys[0]] if eels_keys else next(iter(signals.values()))
for ax in main.axes_manager._axes:
    kind = 'nav' if ax.navigate else 'sig'
    print(f'[{kind}] {ax.name:<12} {ax.size:>5} @ {ax.scale:>10.6g} {ax.units:<3} offset {ax.offset:.6g}')
n_frames = main.axes_manager['Time'].size if is_series(main) else 1
print(f'\n{n_frames} frames')
""")

md(r"""
### Frame alignment (specimen drift)

The drift is measured on the HAADF series (phase correlation, each frame against the one
before by default), rounded to whole pixels of the coarsest stream (the SI), and every stream
is **cropped** to the field common to all frames. Nothing is interpolated, so spectra keep
their counts and noise, and the aligned HAADF still rebins exactly onto the aligned SI. The
price is up to half an SI pixel of residual misregistration, reported below.
""")

code(r"""
frame_report = None
if not align_frames:
    print('align_frames is False: skipping')
elif n_frames < 2:
    print('Only one frame: nothing to align')
else:
    ref_key = drift_image or next((k for k in image_series if 'HAADF' in k.upper()),
                                  image_series[0] if image_series else None)
    if ref_key is None or ref_key not in signals:
        raise ValueError(f'No image series to measure the drift on (drift_image={drift_image!r}, '
                         f'image series: {image_series})')
    others = [k for k, s in signals.items() if k != ref_key and is_series(s)]
    logger.debug(f'measuring drift on {ref_key}, applying to {others}')

    t0 = time.perf_counter()
    reference_before = signals[ref_key]
    aligned = gi.align_frames(reference_before, *[signals[k] for k in others],
                              frame_reference=frame_reference)
    *aligned_signals, frame_report = aligned
    for key, s in zip([ref_key] + others, aligned_signals):
        signals[key] = s
    timings['align_frames_s'] = time.perf_counter() - t0

    print(f"total drift {frame_report['total_drift_nm']:.2f} nm over {n_frames} frames, "
          f"kept {100 * frame_report['field_kept']:.0f}% of the field, "
          f"residual {frame_report['residual_rms_nm']:.3f} nm rms")
    for key in [ref_key] + others:
        print(f'  {key:<28} -> {signals[key]}')
""")

code(r"""
if frame_report is not None:
    t_axis = signals[ref_key].axes_manager['Time'].axis
    drift_nm = frame_report['drift_nm']
    applied_nm = frame_report['applied_shifts_coarse_px'] * frame_report['coarse_factor'] \
        * signals[ref_key].axes_manager['x'].scale

    before = reference_before.data
    before = np.asarray(before.compute() if hasattr(before, 'compute') else before)
    after = signals[ref_key].data
    after = np.asarray(after.compute() if hasattr(after, 'compute') else after)

    fig, axs = plt.subplots(1, 4, figsize=(17, 3.8), constrained_layout=True)
    for i, (label, c) in enumerate((('dy', 'C0'), ('dx', 'C1'))):
        axs[0].plot(t_axis, drift_nm[:, i], '.-', c=c, ms=4, label=f'{label} measured')
        axs[0].plot(t_axis, applied_nm[:, i], 's--', c=c, ms=5, mfc='none', lw=0.8, label=f'{label} applied (whole SI px)')
    axs[0].set_xlabel('time (s)'); axs[0].set_ylabel('drift (nm)'); axs[0].legend(fontsize=7)
    axs[0].set_title(f'drift measured on {ref_key}', fontsize=9)
    axs[1].imshow(before.sum(0), cmap='gray'); axs[1].set_title('sum of all frames, before', fontsize=9)
    axs[2].imshow(after.sum(0), cmap='gray'); axs[2].set_title('sum of all frames, aligned', fontsize=9)
    axs[3].imshow(after[0] - after[-1], cmap='RdBu'); axs[3].set_title('aligned: first - last frame', fontsize=9)
    for ax in axs[1:]:
        ax.set_xticks([]); ax.set_yticks([])
    savefig('frame_alignment.png', fig)

    np.savez(diag_dir / 'frame_drift.npz', time_s=t_axis, drift_px=frame_report['drift_px'],
             drift_nm=drift_nm, applied_shifts_coarse_px=frame_report['applied_shifts_coarse_px'],
             residual_px=frame_report['residual_px'])
""")

md(r"""
### Zero-loss peak alignment

Measures the ZLP position at every pixel of every frame on the low loss (it is sharp and
strong) and applies the **same shifts** to the high loss, both staying lazy. Re-zeroes the energy
axis too. It costs a full extra pass over both streams.
""")

code(r"""
zlp_report = None
if not align_zlp:
    print('align_zlp is False: skipping')
elif ll_key is None:
    logger.warning('align_zlp requested but there is no low-loss stream: skipping')
else:
    also = [signals[hl_key]] if hl_key else []
    t0 = time.perf_counter()
    ll_al, *rest = gi.align_to_zlp(signals[ll_key], *also)
    zlp_report = rest[-1]
    signals[ll_key] = ll_al
    if hl_key:
        signals[hl_key] = rest[0]
    timings['align_zlp_s'] = time.perf_counter() - t0
    print(f"ZLP offset {1000 * zlp_report['offset_before_eV']:+.0f} -> {1000 * zlp_report['offset_after_eV']:+.0f} meV, "
          f"per-pixel jitter {1000 * zlp_report['jitter_before_eV']:.0f} -> {1000 * zlp_report['jitter_after_eV']:.0f} meV "
          f"({timings['align_zlp_s']:.0f} s)")
""")

code(r"""
if zlp_report is not None:
    before = zlp_report['shifts_eV']
    after = gi.zlp_position(signals[ll_key])
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.6), constrained_layout=True)
    axs[0].hist(1000 * before.ravel(), bins=100, alpha=0.6, label='before')
    axs[0].hist(1000 * after.ravel(), bins=100, alpha=0.6, label='after')
    axs[0].set_xlabel('ZLP position (meV)'); axs[0].legend(fontsize=8)
    axs[0].set_title('per-pixel ZLP position', fontsize=9)
    t_axis = signals[ll_key].axes_manager['Time'].axis
    axs[1].plot(t_axis, 1000 * np.nanmean(before, axis=(1, 2)), '.-', label='before')
    axs[1].plot(t_axis, 1000 * np.nanmean(after, axis=(1, 2)), '.-', label='after')
    axs[1].set_xlabel('time (s)'); axs[1].set_ylabel('mean ZLP (meV)'); axs[1].legend(fontsize=8)
    axs[1].set_title('frame-to-frame drift', fontsize=9)
    im = axs[2].imshow(1000 * (before[0] - np.nanmean(before[0])), cmap='RdBu')
    fig.colorbar(im, ax=axs[2], label='meV')
    axs[2].set_title('frame 0: ZLP shift applied per pixel', fontsize=9)
    axs[2].set_xticks([]); axs[2].set_yticks([])
    savefig('zlp_alignment.png', fig)
    np.save(diag_dir / 'zlp_position_before_eV.npy', before)
    np.save(diag_dir / 'zlp_position_after_eV.npy', after)
""")

md("### Record the processing in the metadata and save every stream as `.hspy`\n\n`save_rechunked` writes spectrum-contiguous chunks with explicit sizes (HDF5 rejects the dask `-1` shorthand). `.zspy` is not used: it is broken with rosettasciio 0.9 + zarr 3.")

code(r"""
processing = {
    'workflow': 'EELS_insitu_processing',
    'date': datetime.datetime.now().isoformat(timespec='seconds'),
    'source_folder': str(dataset_dir),
    'frames': frames if frames > 0 else 'all',
    'align_frames': align_frames and frame_report is not None,
    'align_zlp': align_zlp and zlp_report is not None,
}
if frame_report is not None:
    processing['frame_alignment'] = {
        'measured_on': ref_key, 'frame_reference': frame_reference,
        'total_drift_nm': frame_report['total_drift_nm'], 'field_kept': frame_report['field_kept'],
        'residual_rms_nm': frame_report['residual_rms_nm'],
    }
if zlp_report is not None:
    processing['zlp_alignment'] = {k: zlp_report[k] for k in
                                   ('offset_before_eV', 'offset_after_eV', 'jitter_before_eV', 'jitter_after_eV')}


def file_name(key):
    return re.sub(r'[^A-Za-z0-9]+', '_', key).strip('_') + '.hspy'


saved = {}
for key, s in signals.items():
    for item, value in processing.items():
        if not isinstance(value, dict):
            s.metadata.set_item(f'Processing.{item}', value)
        else:
            for sub, v in value.items():
                s.metadata.set_item(f'Processing.{item}.{sub}', v)
    path = out_dir / file_name(key)
    t0 = time.perf_counter()
    if s._lazy:
        gi.save_rechunked(s, path, overwrite=True, compression=compression)
    else:
        s.save(path, overwrite=True, compression=compression)
    timings[f'save_{file_name(key)}_s'] = time.perf_counter() - t0
    saved[key] = path
    print(f'{key:<28} -> {path.name}  {path.stat().st_size / 1e9:.3f} GB on disk  ({timings[f"save_{file_name(key)}_s"]:.0f} s)')
    logger.debug(f'saved {key} to {path}')
""")

md("### Quick look at the saved files\n\nRe-reads them from disk, so this also checks they are complete. One pass over each EELS stream.")

code(r"""
if make_quicklook:
    t0 = time.perf_counter()
    reloaded = {k: hs.load(p, lazy=True) for k, p in saved.items()}
    fig = gi.quicklook(reloaded)
    savefig('quicklook.png', fig)
    for s in reloaded.values():
        if s._lazy:
            s.close_file()
    timings['quicklook_s'] = time.perf_counter() - t0
else:
    print('make_quicklook is False: skipping')
""")

md("### Write the analysis notebook\n\nA notebook next to the saved files with code to lazily load them and explore them. It is written, not executed.")

code(r'''
def suggest_window(signal, lo_frac, hi_frac):
    ax = signal.axes_manager.signal_axes[0]
    e0, e1 = ax.offset, ax.offset + ax.scale * (ax.size - 1)
    return (round(e0 + lo_frac * (e1 - e0)), round(e0 + hi_frac * (e1 - e0)))


hl_file = saved.get(hl_key)
ll_file = saved.get(ll_key)
img_key = next((k for k in image_series if 'HAADF' in k.upper()), image_series[0] if image_series else None)
files = {k: p.name for k, p in saved.items()}
window = suggest_window(signals[hl_key], 0.45, 0.55) if hl_key else None
fit_window = suggest_window(signals[hl_key], 0.25, 0.4) if hl_key else None
edge_window = suggest_window(signals[hl_key], 0.42, 0.52) if hl_key else None

steps = []
if processing['align_frames']:
    fa = processing['frame_alignment']
    steps.append(f"frames aligned for drift (measured on {fa['measured_on']}: {fa['total_drift_nm']:.2f} nm "
                 f"total, {100 * fa['field_kept']:.0f}% of the field kept, {fa['residual_rms_nm']:.3f} nm rms residual)")
if processing['align_zlp']:
    za = processing['zlp_alignment']
    steps.append(f"ZLP aligned: offset {1000 * za['offset_before_eV']:+.0f} -> {1000 * za['offset_after_eV']:+.0f} meV, "
                 f"jitter {1000 * za['jitter_before_eV']:.0f} -> {1000 * za['jitter_after_eV']:.0f} meV")
steps = steps or ['calibrated only (no alignment)']

acells = []
amd = lambda s: acells.append(nbformat.v4.new_markdown_cell(s.strip('\n')))
acode = lambda s: acells.append(nbformat.v4.new_code_cell(s.strip('\n')))

amd(f"""
# {dataset_name} — multiframe EELS analysis

Written by the `EELS_insitu_processing` workflow on {processing['date']}.

| | |
|---|---|
| raw data | `{dataset_dir}` |
| frames | {processing['frames']} |
| processing | {'; '.join(steps)} |

The calibrated streams are the `.hspy` files in this folder. Everything below loads them
**lazily**: nothing is read until a cell asks for it.
""")

acode(f"""
%matplotlib widget
# `widget` (ipympl) gives interactive HyperSpy plots; swap for `inline` for static figures.
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs

{'sys.path.insert(0, ' + repr(str(gatan_insitu_path)) + ')' if str(gatan_insitu_path).strip() else '# sys.path.insert(0, "<folder containing gatan_insitu>")  # if it is not installed'}
import gatan_insitu as gi

OUT = Path({str(out_dir)!r})          # <- this folder; change it if the files move
FILES = {files!r}
""")

amd("## 1. Load the saved signals, lazily")

acode(f"""
signals = {{key: hs.load(OUT / name, lazy=True) for key, name in FILES.items()}}
for key, s in signals.items():
    print(f"{{key:<28}} {{s}}")

{('hl = signals[' + repr(hl_key) + ']      # high loss') if hl_key else 'hl = None'}
{('ll = signals[' + repr(ll_key) + ']      # low loss') if ll_key else 'll = None'}
{('haadf = signals[' + repr(img_key) + ']') if img_key else 'haadf = None'}
""")

if hl_key:
    acode("""
# Fully calibrated: time from the recorded frame timestamps, space and energy from the tags.
for ax in hl.axes_manager._axes:
    kind = "nav" if ax.navigate else "sig"
    print(f"[{kind}] {ax.name:<12} {ax.size:>5} @ {ax.scale:>10.6g} {ax.units:<3} offset {ax.offset:.6g}")
print(f"\\narray shape {hl.data.shape}  (time, y, x, E),  chunks {hl.data.chunksize}")
""")

amd("""
### Axis order: the thing that trips everyone up

HyperSpy prints *navigation* axes in **reverse** array order: `hl` shows `(x, y, time | E)`
while `hl.data.shape` is `(time, y, x, E)`, and `hl.inav[i, j, k]` is `[x, y, time]`. So
`axes_manager[0]` is **x**, not time — index by name instead.
""")

if hl_key:
    acode("""
print("axes_manager['Time'] ->", hl.axes_manager["Time"].scale, "s per frame")
t = hl.metadata.get_item("InSitu.frame_times_s")
if t is not None:
    print("timestamps:", np.round(t[:4], 2), "...", np.round(t[-1], 2), "s")

# What the workflow did to these files:
hl.metadata.Processing
""")
    acode("""
# Microscope parameters come across from the DM tags, so EELS routines find them set.
hl.metadata.Acquisition_instrument
""")

amd("""
## 2. First look

Survey, first/last image frame, summed spectra and the intensity trend, in one pass over each
stream. (The workflow saved the same figure as `diagnostics/quicklook.png`.)
""")
acode("fig = gi.quicklook(signals)")

if hl_key:
    amd("""
## 3. Interactive browsing

For a lazy signal `plot()` builds its navigator from a single chunk, which is slow and shows
only part of the spectrum. Give it a real one: the simultaneous HAADF (rebinned exactly onto
the SI grid) or an energy-window map.
""")
    nav_src = 'haadf=haadf' if img_key else f'energy_window={window!r}'
    acode(f"""
nav = gi.make_navigator(hl, {nav_src})
hl.navigator = nav          # reused by every later hl.plot()
hl.plot()                   # drag the navigator; the spectrum follows
""")

    amd("""
## 4. Energy-window maps (cheap)

A narrow window only reads the chunks that cover it. **Edit `WINDOW`** to your edge — the
default is just the middle of the recorded range.
""")
    acode(f"""
WINDOW = {window!r}      # eV, e.g. (840., 900.) for Ni L2,3

t0 = time.perf_counter()
emap = gi.computed(hl.isig[float(WINDOW[0]):float(WINDOW[1])].sum(axis="Energy loss"))
print(f"map of {{WINDOW}} eV, all frames: {{time.perf_counter()-t0:.1f}} s -> {{emap.data.shape}}")

n = emap.data.shape[0]
show = sorted(set(np.linspace(0, n - 1, min(4, n)).astype(int)))
times = hl.axes_manager["Time"].axis
fig, axes = plt.subplots(1, len(show), figsize=(3 * len(show), 3.2), constrained_layout=True, squeeze=False)
for ax, k in zip(axes[0], show):
    ax.imshow(emap.data[k], cmap="inferno")
    ax.set_title(f"frame {{k}}  ({{times[k]:.0f}} s)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"{{WINDOW[0]}}-{{WINDOW[1]}} eV through the series", fontsize=10);
""")
    amd("**`compute()` works in place and returns `None`** — `gi.computed(...)` computes *and* returns the signal.")

if ll_key:
    amd(f"""
## 5. Low loss: zero-loss peak

The ZLP *peak* position at every pixel (refined parabolically over ±1.5 eV — never a centroid,
which the plasmon tail drags around). {'The workflow aligned on it, so this shows the residual.' if processing['align_zlp'] else 'Not aligned by the workflow: `gi.align_to_zlp(ll, hl)` would correct it (reads both streams twice).'}
""")
    acode("""
pos = gi.zlp_position(ll)          # (frames, y, x), eV
disp = ll.axes_manager["Energy loss"].scale
print(f"ZLP offset from zero  : {1000*np.nanmean(pos):+7.1f} meV")
print(f"per-pixel jitter (sd) : {1000*np.nanstd(pos):7.1f} meV = {np.nanstd(pos)/disp:.2f} channels")
print(f"within-frame scatter  : {1000*np.nanmean(np.nanstd(pos, axis=(1,2))):7.1f} meV")
print(f"frame-to-frame drift  : {1000*np.ptp(np.nanmean(pos, axis=(1,2))):7.1f} meV p2p")

fig, ax = plt.subplots(figsize=(6.5, 3.2), constrained_layout=True)
ax.plot(ll.axes_manager["Time"].axis, 1000 * np.nanmean(pos, axis=(1, 2)), ".-", ms=4)
ax.set_xlabel("time (s)"); ax.set_ylabel("mean ZLP position (meV)"); ax.grid(alpha=.3);
""")

if hl_key:
    amd("""
## 6. How the whole field changes through the series

The mean spectrum per frame (one pass over the stream), first against last frame, and a
background-subtracted edge integral per frame with a significance test. **Edit the windows**:
`EDGE` starts at the edge onset, `FIT` is an edge-free pre-edge range for the power law.
""")
    acode("""
field = gi.computed(hl.mean(axis=["y", "x"]))          # (Time | Energy loss)
e = field.axes_manager["Energy loss"].axis
times = field.axes_manager["Time"].axis

fig, ax = plt.subplots(figsize=(8, 3.4), constrained_layout=True)
ax.plot(e, field.data[0], lw=0.8, label=f"frame 0 ({times[0]:.0f} s)")
ax.plot(e, field.data[-1], lw=0.8, label=f"frame {len(times)-1} ({times[-1]:.0f} s)")
ax.set_xlabel("energy loss (eV)"); ax.set_ylabel("counts / pixel"); ax.set_yscale("log")
ax.legend(fontsize=8); ax.set_title("mean spectrum, first vs last frame", fontsize=9);
""")
    acode(f"""
EDGE = {edge_window!r}      # integration window, eV — edit to your edge
FIT = {fit_window!r}       # pre-edge power-law fit window, eV — must contain no edge

result = gi.edge_intensity({{"field": field}}, EDGE, FIT)["field"]
t = gi.trend(result["intensity"], times)
if t["n"] >= 3:
    print(f"edge {{EDGE}} eV: mean {{t['mean']:.4g}}, change {{100*t['change']:+.1f}}% over the series, p = {{t['p']:.3f}}")
else:
    print(f"edge {{EDGE}} eV: only {{t['n']}} usable frames, too few for a trend")

fig, ax = plt.subplots(figsize=(6.5, 3.2), constrained_layout=True)
ax.errorbar(times, result["intensity"], yerr=result["error"], fmt=".-", ms=4)
ax.set_xlabel("time (s)"); ax.set_ylabel("edge integral"); ax.grid(alpha=.3)
ax.set_title(f"{{EDGE[0]}}-{{EDGE[1]}} eV, background fitted over {{FIT[0]}}-{{FIT[1]}} eV", fontsize=9);
""")

amd("""
## 7. Further analysis with `gatan_insitu`

| task | function |
|---|---|
| bin frames to trade time resolution for signal | `gi.bin_frames(spectra, n, times=...)` |
| moving-window averages (pass `effective_n` to `gi.trend`!) | `gi.rolling_frames(spectra, w, times=...)` |
| edge integrals for many edges at once | `gi.edge_table(spectra, times=...)` |
| L3/L2 white-line ratio — **fit, don't integrate** | `gi.hyperspy_white_lines(spectra)`, `gi.fit_white_lines` |
| a thickness-insensitive edge map | `gi.jump_ratio_map(hl, pre=..., post=...)` |
| elemental map with background removal | `gi.elemental_map(hl, window, background)` |
| a small, portable copy of a subset | `gi.save_rechunked(hl.isig[700.:1000.], "subset.hspy")` |

`spectra` here is a dict of `(Time | Energy loss)` signals, e.g. `{"field": field}` from section 6.

## Cheat sheet

| Gotcha | What to do |
|---|---|
| `axes_manager[0]` is `x`, not `Time` | index by name: `axes_manager["Time"]` |
| `s.compute()` returns `None` | `gi.computed(s)` |
| `plot()` navigator shows one chunk | `gi.make_navigator(s, haadf=haadf)` |
| lazy `.hspy` won't delete/move on Windows | `s.close_file()` first (last cell) |
| `.zspy` fails in this env | use `.hspy` |
""")
acode("""
# Release the files (needed on Windows before moving or deleting them)
for s in signals.values():
    if s._lazy:
        s.close_file()
""")

analysis_nb = nbformat.v4.new_notebook(cells=acells)
analysis_nb.metadata = {'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
                        'language_info': {'name': 'python'}}
nbformat.validate(analysis_nb)
safe_name = re.sub(r'[^A-Za-z0-9]+', '_', dataset_name).strip('_') or 'dataset'
analysis_path = out_dir / f'{safe_name}_analysis.ipynb'
nbformat.write(analysis_nb, analysis_path)
print('analysis notebook:', analysis_path)
''')

md("### Write a summary for provenance")

code(r"""
def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist() if x.size <= 1000 else f'array{list(x.shape)} (see diagnostics/)'
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


summary = {
    **processing,
    'visit': visit, 'sample': sample, 'dataset_name': dataset_name,
    'output_folder': out_dir, 'files': {k: p.name for k, p in saved.items()},
    'analysis_notebook': analysis_path.name,
    'shapes': {k: list(s.data.shape) for k, s in signals.items()},
    'frame_alignment_report': {k: v for k, v in (frame_report or {}).items()},
    'zlp_alignment_report': {k: v for k, v in (zlp_report or {}).items() if k != 'shifts_eV'},
    'timings_s': timings,
    'versions': {'hyperspy': hs.__version__, 'gatan_insitu': gi.__version__},
}
with open(out_dir / 'processing_summary.json', 'w') as fp:
    json.dump(jsonable(summary), fp, indent=2)
logger.debug(f'Finished. Results in {out_dir}')
print('Results written to', out_dir)
""")

nb = new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbformat.validate(nb)
nbformat.write(nb, sys.argv[1])
print("wrote", sys.argv[1], len(cells), "cells")
