import os
import glob
import shutil
import subprocess
import numpy as np
import pandas as pd
import nibabel as nib
import ants

from scipy import ndimage
IXI_DIR = "path/to/archive"
SYNTHSEG_DIR = "path/to/synthseg_freesurfer/SynthSeg"
MNI305_TEMPLATE = "path/to/average305_t1_tal_lin.nii"

ONTOLOGY_CSV = "path/to/canonical_brain_ontology_v2_FINAL.csv"
BATCH_OUTPUT = "./IXI_SYNTHSEG_BATCH"

SYNTHSEG_SCRIPT = "path/to/synthseg_freesurfer/SynthSeg/scripts/commands/SynthSeg_predict.py"


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
# ONTOLOGY
# ============================================================

ontology = pd.read_csv(
    ONTOLOGY_CSV
)

ontology_lookup = ontology.set_index(
    "canonical_id"
).to_dict("index")


# ============================================================
# MNI305
# ============================================================

fixed = ants.image_read(
    MNI305_TEMPLATE
)

print(
    "MNI305:",
    fixed.shape,
    fixed.spacing
)


# ============================================================
# FIND FILES
# ============================================================

raw_paths = (
    glob.glob(os.path.join(IXI_DIR, "**", "*.nii"), recursive=True) +
    glob.glob(os.path.join(IXI_DIR, "**", "*.nii.gz"), recursive=True)
)

# Filter to keep ONLY actual files (ignoring directories named .nii) and take the first 50
files = sorted([f for f in raw_paths if os.path.isfile(f)])[:50]

print(
    "Files found:",
    len(files)
)


# ============================================================
# PROCESS
# ============================================================

def process_subject(t1_path):

    filename = os.path.basename(
        t1_path
    )

    subject_id = filename.split("-")[0]

    print("\n" + "=" * 70)
    print("SUBJECT:", subject_id)
    print("=" * 70)

    subject_dir = os.path.join(
        BATCH_OUTPUT,
        subject_id
    )

    ras_dir = os.path.join(
        subject_dir,
        "RAS"
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

    os.makedirs(
        ras_dir,
        exist_ok=True
    )

    os.makedirs(
        synthseg_dir,
        exist_ok=True
    )

    os.makedirs(
        mni_dir,
        exist_ok=True
    )

    os.makedirs(
        transform_dir,
        exist_ok=True
    )

    ras_path = os.path.join(
        ras_dir,
        f"{subject_id}_RAS.nii.gz"
    )

    registered_path = os.path.join(
        mni_dir,
        f"{subject_id}_T1_MNI305.nii.gz"
    )

    transform_path = os.path.join(
        transform_dir,
        f"{subject_id}_to_MNI305_Affine.mat"
    )

    mni_seg_path = os.path.join(
        mni_dir,
        f"{subject_id}_SynthSeg_segmentation_MNI305.nii.gz"
    )

    measurement_path = os.path.join(
        mni_dir,
        f"{subject_id}_SynthSeg_structure_measurements.csv"
    )

    try:

        # ----------------------------------------------------
        # 1. RAS
        # ----------------------------------------------------

        img = nib.load(
            t1_path
        )

        ras_img = nib.as_closest_canonical(
            img
        )

        nib.save(
            ras_img,
            ras_path
        )

        print(
            "RAS:",
            ras_img.shape
        )

        # ----------------------------------------------------
        # 2. SynthSeg
        # ----------------------------------------------------

        print(
            "Running SynthSeg..."
        )

        subprocess.run(
            [
                "python",
                SYNTHSEG_SCRIPT,
                "--i",
                ras_path,
                "--o",
                synthseg_dir,
                '--threads', '4',
                "--cpu",

            ],
            check=True
        )

        # ----------------------------------------------------
        # Find generated SynthSeg file
        # ----------------------------------------------------

        synthseg_files = glob.glob(
            os.path.join(
                synthseg_dir,
                "*.nii"
            )
        )

        synthseg_files += glob.glob(
            os.path.join(
                synthseg_dir,
                "*.nii.gz"
            )
        )

        if not synthseg_files:

            raise RuntimeError(
                "SynthSeg output not found."
            )

        synthseg_path = synthseg_files[0]

        print(
            "SynthSeg output:",
            synthseg_path
        )

        synth_img = nib.load(
            synthseg_path
        )

        synthseg = synth_img.get_fdata().astype(
            np.int16
        )

        print(
            "SynthSeg shape:",
            synthseg.shape
        )

        print(
            "Labels:",
            np.unique(synthseg)
        )

        # ----------------------------------------------------
        # 3. T1 → MNI305
        # ----------------------------------------------------

        print(
            "Registering T1 → MNI305..."
        )

        moving = ants.image_read(
            ras_path
        )

        registration = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="Affine"
        )

        ants.image_write(
            registration["warpedmovout"],
            registered_path
        )

        shutil.copy(
            registration["fwdtransforms"][0],
            transform_path
        )

        print(
            "Transform:",
            transform_path
        )

        # ----------------------------------------------------
        # 4. Segmentation → MNI305
        # ----------------------------------------------------

        print(
            "Warping segmentation → MNI305..."
        )

        synth_ants = ants.image_read(
            synthseg_path
        )

        warped_seg = ants.apply_transforms(
            fixed=fixed,
            moving=synth_ants,
            transformlist=[
                registration["fwdtransforms"][0]
            ],
            interpolator="nearestNeighbor"
        )

        ants.image_write(
            warped_seg,
            mni_seg_path
        )

        mni_seg = warped_seg.numpy().astype(
            np.int16
        )

        print(
            "MNI305 segmentation:",
            mni_seg.shape
        )

        # ----------------------------------------------------
        # 5. Measurements
        # ----------------------------------------------------

        rows = []

        voxel_volume = np.prod(
            fixed.spacing
        )

        for source_id in sorted(
            np.unique(mni_seg)
        ):

            source_id = int(source_id)

            if source_id == 0:
                continue

            # CSF currently excluded
            if source_id == 24:
                continue

            if source_id not in mapping:
                continue

            canonical_id, laterality = mapping[
                source_id
            ]

            mask = (
                mni_seg == source_id
            )

            cc, n = ndimage.label(
                mask,
                structure=np.ones(
                    (3, 3, 3)
                )
            )

            if n > 1:

                counts = np.bincount(
                    cc.ravel()
                )

                counts[0] = 0

                mask = (
                    cc == counts.argmax()
                )

            voxel_count = int(
                mask.sum()
            )

            if voxel_count < 20:
                continue

            volume_mm3 = float(
                voxel_count *
                voxel_volume
            )

            concept = ontology_lookup.get(
                canonical_id,
                {}
            )

            rows.append({

                "subject_id": subject_id,

                "model": "SynthSeg",

                "model_version": "2.0",

                "canonical_id": canonical_id,

                "preferred_name": concept.get(
                    "preferred_name",
                    ""
                ),

                "laterality": laterality,

                "voxel_count": voxel_count,

                "volume_mm3": volume_mm3,

                "parent_id": concept.get(
                    "parent_id",
                    ""
                ),

                "category": concept.get(
                    "category",
                    ""
                ),

                "ontology_level": concept.get(
                    "ontology_level",
                    ""
                )
            })

        analysis_df = pd.DataFrame(
            rows
        )

        analysis_df.to_csv(
            measurement_path,
            index=False
        )

        print(
            "Structures:",
            len(analysis_df)
        )

        print(
            "Saved:",
            measurement_path
        )

        # ----------------------------------------------------
        # 6. DELETE TEMPORARY FILES
        # ----------------------------------------------------

        if os.path.exists(
            ras_dir
        ):
            shutil.rmtree(
                ras_dir
            )

        if os.path.exists(
            synthseg_dir
        ):
            shutil.rmtree(
                synthseg_dir
            )

        print(
            "Temporary RAS + SynthSeg files deleted."
        )

        return True

    except Exception as e:

        print(
            f"FAILED {subject_id}:",
            repr(e)
        )

        print(
            "Temporary files retained for debugging."
        )

        return False


# ============================================================
# RUN
# ============================================================

os.makedirs(
    BATCH_OUTPUT,
    exist_ok=True
)

results = []

for i, f in enumerate(files, 1):

    subject_id = os.path.basename(f).split("-")[0]

    measurement_path = os.path.join(
        BATCH_OUTPUT,
        subject_id,
        "MNI305",
        f"{subject_id}_SynthSeg_structure_measurements.csv"
    )

    if os.path.exists(measurement_path):
        print(f"[{i}/{len(files)}] SKIP {subject_id} — already completed")
        continue

    print(f"\n[{i}/{len(files)}]")

    success = process_subject(f)

    results.append({
        "file": f,
        "success": success
    })

# ============================================================
# COMBINE
# ============================================================

measurement_files = glob.glob(
    os.path.join(
        BATCH_OUTPUT,
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

    combined_path = os.path.join(
        BATCH_OUTPUT,
        "SynthSeg_structure_measurements_ALL.csv"
    )

    all_measurements.to_csv(
        combined_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("BATCH COMPLETE")
    print("=" * 70)

    print(
        "Successful:",
        sum(x["success"] for x in results)
    )

    print(
        "Subjects:",
        all_measurements["subject_id"].nunique()
    )

    print(
        "Rows:",
        len(all_measurements)
    )

    print(
        "Unique structures:",
        all_measurements["canonical_id"].nunique()
    )

    print(
        "Saved:",
        combined_path
    )