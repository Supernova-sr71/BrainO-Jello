# BrainO-Jello

A research toolkit for converting structural brain MRI segmentations into an ontology-linked 3D brain atlas.

BrainO-Jello brings together MRI preprocessing, spatial normalization, automated segmentation, anatomical ontology mapping, volumetric analysis, mesh generation, and interactive 3D visualization. The repository currently includes workflows based on MONAI UNesT and SynthSeg outputs, with results represented in MNI305/RAS space and exported as CSV, GLB, and JSON artifacts.

> **Project status:** Research / experimental. Several scripts are configured for local or Kaggle environments and may require path and model-configuration changes before they run on another machine.

## Highlights

- Reorient T1-weighted MRI volumes to RAS orientation.
- Register subject images and segmentations to the MNI305 template.
- Run or consume brain segmentations from MONAI UNesT and SynthSeg.
- Map source labels to a canonical brain ontology with laterality metadata.
- Generate per-structure meshes and combined `.glb` brain models.
- Produce structure-level measurements, agreement tables, and statistical summaries.
- Explore canonical anatomical structures in a browser-based Three.js viewer.

## Repository structure

```text
.
├── IXI351_UNEST_CanonicalBrain_CLEAN.glb   # Example canonical brain GLB asset
├── Ontology/                               # Ontology-related files and resources
├── brain_viewer_data/                      # Viewer assets such as GLB and manifest files
├── canonical_3d_brain_viewer/              # Packaged 3D viewer output
├── code/                                   # Reusable / path-portable script variants
├── optimized_meshes/                        # Optimized anatomical meshes
├── synthseg_freesurfer/                     # SynthSeg-related resources
├── batchprocess_t1.py                       # Batch T1 → UNesT → mesh pipeline
├── complete-glb_pipeline.py                # Single-subject end-to-end GLB pipeline
├── glb_viewer.py                            # Generates the browser-based 3D viewer
├── synthseg_test_script.py                  # SynthSeg batch-processing workflow
├── statistics_data.py                       # Agreement and correlation statistics
├── statistics_data2.ipynb                   # Exploratory statistical notebook
├── synthseg_labels_table.txt                # SynthSeg label reference
├── requirements.txt                          # Python dependencies
└── *.csv                                    # Measurements, mappings, and agreement data
```

## Processing overview

The main processing workflow follows this sequence:

```text
T1 MRI
  ↓
RAS reorientation
  ↓
Affine registration to MNI305
  ↓
UNesT or SynthSeg segmentation
  ↓
Source-label → canonical-ontology mapping
  ↓
Structure measurements and connected-component cleanup
  ↓
Marching-cubes mesh extraction
  ↓
Per-structure GLB files + combined brain.glb
  ↓
Manifest and interactive 3D visualization
```

Generated manifests describe the model, coordinate system, ontology version, registration transform, and available anatomical structures. Each structure can include its canonical ID, preferred name, source label, laterality, category, mesh filename, volume, and mesh-quality metadata.

## Requirements

The repository targets Python-based neuroimaging workflows. The core dependencies currently listed in `requirements.txt` are:

- MONAI
- NumPy `<2.0.0`
- ANTsPyX
- trimesh
- fast-simplification

The pipelines also use packages such as `pandas`, `nibabel`, `scipy`, `scikit-image`, `statsmodels`, and PyTorch. Depending on the selected workflow, SynthSeg and a compatible pretrained model may also be required.

A GPU is recommended for UNesT inference, although the scripts select CPU execution when CUDA is unavailable.

## Installation

Create an isolated environment and install the declared dependencies:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For the statistical scripts, install any additional packages that are not already provided by the selected MONAI environment:

```bash
pip install pandas nibabel scipy scikit-image statsmodels torch
```

## Data and model prerequisites

Before running a pipeline, provide the following inputs and update the configuration paths in the relevant script:

1. One or more T1-weighted NIfTI files (`.nii` or `.nii.gz`).
2. An MNI305 template, such as `average305_t1_tal_lin.nii`.
3. The canonical ontology CSV.
4. A source-label mapping CSV for the selected segmentation model.
5. A compatible UNesT model and model-loading code for the MONAI workflow, or a local SynthSeg installation for the SynthSeg workflow.

The checked-in scripts contain example paths for Kaggle and local development environments. Replace those paths with paths valid for your system before execution.

## Usage

### Run the UNesT batch pipeline

`batchprocess_t1.py` scans an IXI directory, processes each T1 image, registers it to MNI305, performs UNesT inference, creates canonical meshes, and writes a batch summary.

Update the configuration section first, including:

- `IXI_DIR`
- `MNI305_TEMPLATE`
- `ONTOLOGY_CSV`
- `MAPPING_CSV`
- `BATCH_OUTPUT`
- UNesT model initialization

Then run:

```bash
python batchprocess_t1.py
```

Typical per-subject outputs include:

```text
<subject>/MNI305/
├── <subject>_T1_MNI305.nii.gz
├── <subject>_UNEST_segmentation_MNI305.nii.gz
├── transforms/<subject>_to_MNI305_Affine.mat
├── meshes/*.glb
├── mesh_registry.csv
├── brain.glb
└── manifest.json
```

### Run the single-subject GLB pipeline

`complete-glb_pipeline.py` demonstrates the end-to-end process for one subject:

```bash
python complete-glb_pipeline.py
```

This script is useful for validating the full transformation from a T1 volume to a canonical, ontology-linked GLB scene before launching a larger batch run.

### Run the SynthSeg workflow

Configure `synthseg_test_script.py` or the more portable version under `code/` with paths to:

- the IXI dataset,
- the SynthSeg installation,
- the MNI305 template, and
- the canonical ontology.

Then run:

```bash
python synthseg_test_script.py
```

The workflow reorients each subject, invokes SynthSeg, registers the segmentation to MNI305 using nearest-neighbor interpolation, and writes canonical structure measurements.

### Generate the 3D viewer

`glb_viewer.py` prepares a self-contained viewer directory containing:

- `brain.glb`,
- `manifest.json`, and
- `index.html`.

After correcting the paths near the top of the script, run:

```bash
python glb_viewer.py
```

Serve the generated viewer directory through a local HTTP server. Opening the HTML file directly may prevent browser asset loading because of local file security restrictions.

```bash
cd brain_viewer_data
python -m http.server 8000
```

Open `http://localhost:8000` in a browser. The viewer supports:

- orbit rotation and zoom,
- clicking anatomical structures,
- searching by name, canonical ID, or laterality,
- displaying structure metadata,
- isolating a selected structure,
- hiding structures, and
- resetting the scene.

The viewer imports Three.js from a CDN, so an internet connection is required unless those assets are vendored locally.

### Run statistical analysis

The statistics workflow compares UNesT and SynthSeg structure volumes using measures such as:

- ICC(2,1),
- Pearson correlation,
- Spearman correlation,
- Bland–Altman bias and limits of agreement,
- proportional-bias regression, and
- Benjamini–Hochberg FDR-adjusted values.

Set the `INPUT` and `OUT` paths in `statistics_data.py`, then run:

```bash
python statistics_data.py
```

The repository also includes CSV outputs and a notebook for exploratory analysis.

## Data products

The repository contains several types of derived data:

| Format | Purpose |
| --- | --- |
| `.nii` / `.nii.gz` | MRI volumes, registered images, and segmentation labels |
| `.csv` | Ontology mappings, structure measurements, agreement tables, and statistics |
| `.glb` | Per-structure meshes and combined 3D brain scenes |
| `.json` | Viewer and pipeline manifests |
| `.ipynb` | Interactive analysis and visualization |
| `.txt` | Label tables and reference mappings |

## Coordinate systems and ontology

The pipelines use RAS-oriented images and MNI305 registration as the common spatial reference. Segmentation labels are translated into canonical identifiers so that structures produced by different models can be compared using shared names, hierarchy, category, and laterality fields.

Check the ontology and mapping files before comparing results across models. Differences in label definitions, excluded structures, connected-component filtering, voxel spacing, and registration settings can affect volume and mesh measurements.

## Reproducibility notes

- Replace all machine-specific absolute paths before running scripts.
- Record the segmentation model version and checkpoint used for each experiment.
- Keep the MNI305 template, ontology version, mapping version, and preprocessing settings with generated outputs.
- Use nearest-neighbor interpolation for discrete segmentation labels during spatial transformation.
- Verify image orientation, affine matrices, voxel spacing, and label IDs before interpreting measurements.
- Large NIfTI and GLB files can require substantial storage and memory.
- The repository currently contains research scripts rather than a fully packaged command-line application.

## Limitations

- The current scripts are not yet fully configuration-driven.
- Some workflows depend on external datasets, templates, model weights, or SynthSeg source code that are not bundled with this repository.
- Several scripts assume Kaggle or a specific local directory layout.
- The generated anatomical meshes and measurements should be validated for the intended scientific use case.
- This project is not a clinical diagnostic tool and should not be used for medical decision-making.

## Contributing

Contributions are welcome. When proposing changes:

1. Explain the scientific or engineering motivation.
2. Include the input assumptions and expected output format.
3. Avoid committing private patient data, model weights, or large generated artifacts unless explicitly intended.
4. Add or update documentation for new scripts, ontology fields, or pipeline stages.
5. Report the dataset, model version, template, and environment used to reproduce results.

## License

No license file is currently included. Until a license is added, all rights to the repository contents remain with the copyright holder, and reuse should be treated as requiring permission.

## Citation and attribution

If you use this repository in research, cite the underlying resources and methods used in your pipeline, including the selected segmentation model, MNI305 template, ontology, IXI dataset, SynthSeg, MONAI, ANTs, and any other external resources. Add a project citation here once a preferred publication or citation format is established.
