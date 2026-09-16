import os
import glob
import subprocess
import numpy as np
import pandas as pd
import nibabel as nib
import ants

IXI_DIR = "/path/to/ixi-t1-basemri"
SYNTHSEG_DIR = "/Users/abhinavkrishna/githubzz/Brain-O-jello/synthseg_freesurfer/SynthSeg"
MNI305_TEMPLATE = "/Users/abhinavkrishna/githubzz/Brain-O-jello/average305_t1_tal_lin.nii"

ONTOLOGY_CSV = "/Users/abhinavkrishna/githubzz/Brain-O-jello/canonical_brain_ontology_v2_FINAL.csv"

OUTPUT_DIR = "./IXI_SYNTHSEG_BATCH"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SYNTHSEG LABEL → CANONICAL
# ============================================================

mapping = {
    2:  ("BRN-CER-WM", "LEFT"),
    3:  ("BRN-CER-COR", "LEFT"),
    4:  ("BRN-LAT-V", "LEFT"),
    5:  ("BRN-INF-LAT-V", "LEFT"),
    7:  ("BRN-CB-WM", "LEFT"),
    8:  ("BRN-CB-EXT", "LEFT"),
    10: ("BRN-THA", "LEFT"),
    11: ("BRN-CAU", "LEFT"),
    12: ("BRN-PUT", "LEFT"),
    13: ("BRN-PAL", "LEFT"),
    14: ("BRN-3RD-VENTRICLE", "NONE"),
    15: ("BRN-4TH-VENTRICLE", "NONE"),
    16: ("BRN-BST", "NONE"),
    17: ("BRN-HIP", "LEFT"),
    18: ("BRN-AMY", "LEFT"),
    26: ("BRN-ACC", "LEFT"),
    28: ("BRN-VDC", "LEFT"),

    41: ("BRN-CER-WM", "RIGHT"),
    42: ("BRN-CER-COR", "RIGHT"),
    43: ("BRN-LAT-V", "RIGHT"),
    44: ("BRN-INF-LAT-V", "RIGHT"),
    46: ("BRN-CB-WM", "RIGHT"),
    47: ("BRN-CB-EXT", "RIGHT"),
    49: ("BRN-THA", "RIGHT"),
    50: ("BRN-CAU", "RIGHT"),
    51: ("BRN-PUT", "RIGHT"),
    52: ("BRN-PAL", "RIGHT"),
    53: ("BRN-HIP", "RIGHT"),
    54: ("BRN-AMY", "RIGHT"),
    58: ("BRN-ACC", "RIGHT"),
    60: ("BRN-VDC", "RIGHT"),
}


# ============================================================
# LOAD ONTOLOGY
# ============================================================

ontology = pd.read_csv(ONTOLOGY_CSV)

ontology_lookup = (
    ontology
    .drop_duplicates("canonical_id")
    .set_index("canonical_id")
)


# ============================================================
# FIND IXI FILES
# ============================================================

# files = glob.glob(
#     os.path.join(IXI_DIR, "**", "*.nii"),
#     recursive=True
# )

# files += glob.glob(
#     os.path.join(IXI_DIR, "**", "*.nii.gz"),
#     recursive=True
# )
files=['/Users/abhinavkrishna/Downloads/IXI314-IOP-0889-T1.nii.gz']
files = sorted(set(files))

print("Found:", len(files), "MRI files")


# ============================================================
# SYNTHSEG COMMAND
# ============================================================

synthseg_script = os.path.join(
    SYNTHSEG_DIR,
    "scripts",
    "commands",
    "SynthSeg_predict.py"
)


# ============================================================
# PROCESS ONE SUBJECT
# ============================================================

def process_subject(t1_path):

    filename = os.path.basename(t1_path)

    # Example:
    # IXI351-Guys-0914...
    subject_id = filename.split("-")[0]

    print("\n" + "=" * 70)
    print("SUBJECT:", subject_id)
    print("=" * 70)

    subject_dir = os.path.join(
        OUTPUT_DIR,
        subject_id
    )

    synthseg_dir = os.path.join(
        subject_dir,
        "SynthSeg"
    )

    mni_dir = os.path.join(
        subject_dir,
        "MNI305"
    )

    transform_dir = os.path.join(
        mni_dir,
        "transforms"
    )

    os.makedirs(synthseg_dir, exist_ok=True)
    os.makedirs(mni_dir, exist_ok=True)
    os.makedirs(transform_dir, exist_ok=True)


    # --------------------------------------------------------
    # 1. RAS T1
    # --------------------------------------------------------

    ras_path = os.path.join(
        subject_dir,
        f"{subject_id}_RAS.nii.gz"
    )

    img = nib.load(t1_path)
    ras = nib.as_closest_canonical(img)

    nib.save(ras, ras_path)

    print("RAS:", ras.shape)


    # --------------------------------------------------------
    # 2. RUN SYNTHSEG
    # --------------------------------------------------------

    synthseg_output = os.path.join(
        synthseg_dir,
        f"{subject_id}_synthseg.nii"
    )
    synthseg_dir = os.path.join(subject_dir, "SynthSeg")
    os.makedirs(synthseg_dir, exist_ok=True)

    synthseg_path = os.path.join(
        synthseg_dir,
        f"{subject_id}_RAS_synthseg.nii.gz"
    )

    cmd = [
        "python",
        synthseg_script,
        "--i", ras_path,
        "--o", synthseg_path,
        "--cpu"
    ]

    print("Running SynthSeg...")
    subprocess.run(cmd, check=True)

    if not os.path.exists(synthseg_path):
        raise RuntimeError(f"SynthSeg output not found: {synthseg_path}")

    print("SynthSeg output:", synthseg_path)




    # --------------------------------------------------------
    # 3. LOAD SYNTHSEG
    # --------------------------------------------------------

    seg_img = nib.load(synthseg_output)
    seg = seg_img.get_fdata().astype(np.int16)

    print("SynthSeg shape:", seg.shape)
    print(
        "Labels:",
        np.unique(seg).astype(int)
    )


    # --------------------------------------------------------
    # 4. LOAD T1 + MNI305 WITH ANTS
    # --------------------------------------------------------

    moving = ants.image_read(ras_path)
    fixed = ants.image_read(MNI305_TEMPLATE)

    print("Registering T1 → MNI305...")

    registration = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="Affine"
    )

    transform = registration["fwdtransforms"][0]

    print("Transform:", transform)


    # --------------------------------------------------------
    # 5. SAVE TRANSFORM
    # --------------------------------------------------------

    saved_transform = os.path.join(
        transform_dir,
        f"{subject_id}_to_MNI305_Affine.mat"
    )

    if transform != saved_transform:
        import shutil
        shutil.copy2(
            transform,
            saved_transform
        )


    # --------------------------------------------------------
    # 6. RESAMPLE SYNTHSEG → MNI305
    # --------------------------------------------------------

    print("Warping segmentation → MNI305...")

    seg_ants = ants.image_read(
        synthseg_output
    )

    seg_mni = ants.apply_transforms(
        fixed=fixed,
        moving=seg_ants,
        transformlist=[transform],
        interpolator="nearestNeighbor"
    )

    seg_mni_path = os.path.join(
        mni_dir,
        f"{subject_id}_SynthSeg_MNI305.nii.gz"
    )

    ants.image_write(
        seg_mni,
        seg_mni_path
    )

    print(
        "MNI305 segmentation:",
        seg_mni.shape
    )


    # --------------------------------------------------------
    # 7. STRUCTURE VOLUMES
    # --------------------------------------------------------

    records = []

    seg_mni_np = seg_mni.numpy()

    for source_id, (canonical_id, laterality) in mapping.items():

        mask = seg_mni_np == source_id

        voxel_count = int(mask.sum())

        if voxel_count == 0:
            continue

        volume_mm3 = float(
            voxel_count
        )

        if canonical_id not in ontology_lookup.index:
            print(
                "WARNING: missing ontology:",
                canonical_id
            )
            continue

        row = ontology_lookup.loc[
            canonical_id
        ]

        records.append({

            "subject_id":
                subject_id,

            "model":
                "SynthSeg",

            "model_version":
                "2.0",

            "source_id":
                source_id,

            "canonical_id":
                canonical_id,

            "preferred_name":
                row["preferred_name"],

            "laterality":
                laterality,

            "voxel_count":
                voxel_count,

            "volume_mm3":
                volume_mm3,

            "parent_id":
                row["parent_id"],

            "category":
                row["category"],

            "ontology_level":
                row["ontology_level"]
        })


    # --------------------------------------------------------
    # 8. SAVE SUBJECT CSV
    # --------------------------------------------------------

    df = pd.DataFrame(records)

    csv_path = os.path.join(
        mni_dir,
        f"{subject_id}_SynthSeg_structure_measurements.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print(
        "Structures:",
        len(df)
    )

    print(
        "Saved:",
        csv_path
    )


# ============================================================
# RUN BATCH
# ============================================================

for i, t1 in enumerate(files):

    print(
        f"\n[{i+1}/{len(files)}]"
    )

    try:

        process_subject(t1)

    except Exception as e:

        print(
            "FAILED:",
            t1
        )

        print(
            repr(e)
        )


# ============================================================
# COMBINE RESULTS
# ============================================================

measurement_files = glob.glob(
    os.path.join(
        OUTPUT_DIR,
        "*",
        "MNI305",
        "*_SynthSeg_structure_measurements.csv"
    )
)

if measurement_files:

    all_measurements = pd.concat(
        [
            pd.read_csv(f)
            for f in measurement_files
        ],
        ignore_index=True
    )

    output_csv = os.path.join(
        OUTPUT_DIR,
        "SynthSeg_structure_measurements_ALL.csv"
    )

    all_measurements.to_csv(
        output_csv,
        index=False
    )

    print("\n" + "=" * 70)
    print("BATCH COMPLETE")
    print("=" * 70)

    print(
        "Subjects:",
        all_measurements["subject_id"].nunique()
    )

    print(
        "Rows:",
        len(all_measurements)
    )

    print(
        "Structures:",
        all_measurements["canonical_id"].nunique()
    )

    print(
        "Saved:",
        output_csv
    )