# EELS in-situ workflow

Workflows-server notebook for Gatan multiframe (in-situ) STEM/EELS: load lazily and
calibrated, optionally align the frames for drift and the low loss on its ZLP, save every
stream as `.hspy`, and write an analysis notebook next to the results.

| path | what |
|---|---|
| `notebooks/EELS_insitu_processing.ipynb` | the workflow notebook (run by papermill) |
| `templates/eels-insitu.yaml` | Argo `ClusterWorkflowTemplate` |
| `schema/eels_insitu_Schema.json`, `schema/eels_insitu_ui.json` | parameter form |
| `tools/build_eels_insitu_nb.py` | generates the notebook: edit this, then rebuild |
| `gatan_insitu/` | the package the notebook imports (a copy of `local_code/gatan_insitu`) |

`notebooks/`, `templates/` and `schema/` match the layout of an instrument folder in
[imaging-workflows](https://github.com/DiamondLightSource/imaging-workflows) (e.g. `e02/`),
so they can be copied in as they are.

## Rebuild the notebook after editing the generator

```bash
python tools/build_eels_insitu_nb.py notebooks/EELS_insitu_processing.ipynb
```

## Run it by hand

```bash
python -m papermill notebooks/EELS_insitu_processing.ipynb out.ipynb -k python3 \
    -p visit /dls/e02/data/2026/mg12345-1 -p sample sampleX -p dataset_name "InSitu (5)" \
    -p align_frames true -p align_zlp true -p gatan_insitu_path /path/to/workflows_eels
```

Add `-p frames 3` for a quick test on the first three frames.

## Before deploying (TODOs in the YAML)

1. **Image**: it needs hyperspy, exspy, scikit-image, nbformat and papermill; `mib2x` is a placeholder.
2. **Package**: the pod must see `gatan_insitu/`. The template mounts `/dls_sw/e02/software` and
   passes it as `gatan_insitu_path` (the folder that *contains* `gatan_insitu/`).

## Tests

```bash
python -m gatan_insitu.test_alignment      # frame drift alignment, 12 checks
python -m gatan_insitu.test_spectroscopy   # incl. ZLP alignment
```
