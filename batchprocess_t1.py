# ============================================================
# IXI BATCH PIPELINE
# ============================================================

import os, json, shutil, gc
import numpy as np
import pandas as pd
import nibabel as nib
import ants
import torch
import trimesh

from glob import glob
from scipy import ndimage
from skimage.measure import marching_cubes

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    NormalizeIntensityd, EnsureTyped
)
from monai.inferers import SlidingWindowInferer


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

IXI_DIR = "/kaggle/input/datasets/jackontreesandminds/ixi-t1-basemri"
MNI305_TEMPLATE = "/kaggle/input/datasets/jackontreesandminds/mni305-template/average305_t1_tal_lin.nii"
ONTOLOGY_CSV = "/kaggle/input/datasets/jackontreesandminds/required-files/canonical_brain_ontology_v2_FINAL.csv"
MAPPING_CSV = "/kaggle/input/datasets/jackontreesandminds/required-files/source_mapping_MONAI_UNEST_v0.2.7_FINAL.csv"

BATCH_OUTPUT = "/kaggle/working/IXI_UNEST_BATCH"

os.makedirs(BATCH_OUTPUT, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
model.eval()

print("Device:", device)


# ------------------------------------------------------------
# LOAD ONTOLOGY + MAPPING
# ------------------------------------------------------------

ontology = pd.read_csv(ONTOLOGY_CSV)
mapping = pd.read_csv(MAPPING_CSV)

mapping["source_id"] = mapping["source_id"].astype(int)

mapping_lookup = mapping.set_index("source_id").to_dict("index")
ontology_lookup = ontology.set_index("canonical_id").to_dict("index")

print("Ontology concepts:", len(ontology))
print("Source mappings:", len(mapping))


# ------------------------------------------------------------
# MONAI PREPROCESSOR + INFERER
# ------------------------------------------------------------

preprocess = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    NormalizeIntensityd(
        keys=["image"],
        nonzero=True,
        channel_wise=True
    ),
    EnsureTyped(keys=["image"])
])

inferer = SlidingWindowInferer(
    roi_size=(96, 96, 96),
    sw_batch_size=1,
    overlap=0.7
)


# ============================================================
# SINGLE SUBJECT
# ============================================================

def process_subject(input_path, subject_id):

    print("\n" + "=" * 60)
    print("SUBJECT:", subject_id)
    print("=" * 60)

    subject_dir = os.path.join(
        BATCH_OUTPUT,
        subject_id
    )

    ras_dir = os.path.join(subject_dir, "RAS")
    mni_dir = os.path.join(subject_dir, "MNI305")
    transform_dir = os.path.join(mni_dir, "transforms")
    mesh_dir = os.path.join(mni_dir, "meshes")

    for d in [ras_dir, mni_dir, transform_dir, mesh_dir]:
        os.makedirs(d, exist_ok=True)


    # --------------------------------------------------------
    # 1. RAS
    # --------------------------------------------------------

    ras_path = os.path.join(
        ras_dir,
        f"{subject_id}_T1_RAS.nii.gz"
    )

    img = nib.load(input_path)
    ras_img = nib.as_closest_canonical(img)
    nib.save(ras_img, ras_path)

    print("1. RAS:", ras_img.shape)


    # --------------------------------------------------------
    # 2. MNI305 REGISTRATION
    # --------------------------------------------------------

    fixed = ants.image_read(MNI305_TEMPLATE)
    moving = ants.image_read(ras_path)

    registration = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="Affine"
    )

    registered_path = os.path.join(
        mni_dir,
        f"{subject_id}_T1_MNI305.nii.gz"
    )

    ants.image_write(
        registration["warpedmovout"],
        registered_path
    )

    transform_path = os.path.join(
        transform_dir,
        f"{subject_id}_to_MNI305_Affine.mat"
    )

    shutil.copy2(
        registration["fwdtransforms"][0],
        transform_path
    )

    print("2. Registered to MNI305")


    # --------------------------------------------------------
    # 3. UNEST
    # --------------------------------------------------------

    data = preprocess({
        "image": registered_path
    })

    image = data["image"].unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = inferer(image, model)

    labels = (
        logits.argmax(dim=1)[0]
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    print(
        "3. UNesT:",
        len(np.unique(labels)),
        "labels"
    )


    # --------------------------------------------------------
    # 4. SAVE SEGMENTATION
    # --------------------------------------------------------

    reference = nib.load(registered_path)

    segmentation_path = os.path.join(
        mni_dir,
        f"{subject_id}_UNEST_segmentation_MNI305.nii.gz"
    )

    seg_img = nib.Nifti1Image(
        labels,
        reference.affine,
        reference.header.copy()
    )

    seg_img.set_data_dtype(np.uint8)

    nib.save(
        seg_img,
        segmentation_path
    )


    # --------------------------------------------------------
    # 5. MERGE SOURCE LABELS
    #    canonical_id + laterality
    # --------------------------------------------------------

    groups = {}

    for source_id, info in mapping_lookup.items():

        if source_id == 0:
            continue

        canonical_id = info["canonical_id"]
        laterality = info["laterality"]

        key = (
            canonical_id,
            laterality
        )

        if key not in groups:
            groups[key] = []

        groups[key].append(source_id)


    # --------------------------------------------------------
    # 6. CREATE CANONICAL MESHES
    # --------------------------------------------------------

    affine = reference.affine

    registry = []

    for (canonical_id, laterality), source_ids in groups.items():

        # Merge all source labels belonging to this concept
        mask = np.isin(
            labels,
            source_ids
        )

        voxel_count = int(mask.sum())

        if voxel_count < 20:
            continue


        # Largest connected component
        cc, n = ndimage.label(
            mask,
            structure=np.ones((3, 3, 3))
        )

        if n > 1:
            counts = np.bincount(cc.ravel())
            counts[0] = 0
            mask = cc == counts.argmax()


        if mask.sum() < 20:
            continue


        # Marching cubes
        verts, faces, _, _ = marching_cubes(
            mask.astype(np.uint8),
            level=0.5
        )


        # voxel → MNI world coordinates
        verts_world = nib.affines.apply_affine(
            affine,
            verts
        )


        mesh = trimesh.Trimesh(
            vertices=verts_world,
            faces=faces,
            process=True
        )

        _, unique_indices = trimesh.grouping.unique_rows(mesh.faces)
        mesh.update_faces(unique_indices)

        if mesh.volume < 0:
            mesh.invert()


        concept = ontology_lookup.get(
            canonical_id,
            {}
        )

        preferred_name = concept.get(
            "preferred_name",
            canonical_id
        )

        object_name = (
            f"{canonical_id}_{laterality}"
        )

        mesh_file = (
            object_name + ".glb"
        )

        mesh_path = os.path.join(
            mesh_dir,
            mesh_file
        )

        mesh.export(mesh_path)


        registry.append({

            "object_name": object_name,

            "source": "MONAI_UNesT",

            "source_version": "0.2.7",

            "source_ids": ",".join(
                map(str, source_ids)
            ),

            "canonical_id": canonical_id,

            "preferred_name": preferred_name,

            "parent_id": concept.get(
                "parent_id"
            ),

            "category": concept.get(
                "category"
            ),

            "ontology_level": concept.get(
                "ontology_level"
            ),

            "laterality": laterality,

            "mesh_file": mesh_file,

            "voxel_count": voxel_count,

            "vertices": len(mesh.vertices),

            "faces": len(mesh.faces),

            "volume_mm3": abs(
                float(mesh.volume)
            ),

            "watertight": bool(
                mesh.is_watertight
            )
        })


    registry = pd.DataFrame(registry)

    registry_path = os.path.join(
        mni_dir,
        "mesh_registry.csv"
    )

    registry.to_csv(
        registry_path,
        index=False
    )


    # --------------------------------------------------------
    # 7. COMBINED GLB
    # --------------------------------------------------------

    scene = trimesh.Scene()

    for _, row in registry.iterrows():

        mesh_path = os.path.join(
            mesh_dir,
            row["mesh_file"]
        )

        mesh = trimesh.load(
            mesh_path,
            force="mesh"
        )

        scene.add_geometry(
            mesh,
            node_name=row["object_name"],
            geom_name=row["object_name"]
        )


    brain_glb = os.path.join(
        mni_dir,
        "brain.glb"
    )

    scene.export(brain_glb)


    # --------------------------------------------------------
    # 8. MANIFEST
    # --------------------------------------------------------

    manifest = {

        "schema": "1.0",

        "subject_id": subject_id,

        "model": {
            "name": "MONAI Whole Brain Large UNesT",
            "version": "0.2.7",
            "source": "MONAI",
            "label_count": 133
        },

        "space": {
            "name": "MNI305",
            "coordinate_system": "RAS",
            "voxel_spacing_mm": [1, 1, 1]
        },

        "ontology": {
            "name": "Canonical Brain Ontology",
            "version": "2.0"
        },

        "segmentation": os.path.basename(
            segmentation_path
        ),

        "registration": {
            "transform": os.path.relpath(
                transform_path,
                mni_dir
            )
        },

        "structures": registry.to_dict(
            orient="records"
        )
    }


    manifest_path = os.path.join(
        mni_dir,
        "manifest.json"
    )

    with open(manifest_path, "w") as f:
        json.dump(
            manifest,
            f,
            indent=2
        )


    # Free GPU memory
    del image, logits, data
    torch.cuda.empty_cache()
    gc.collect()


    print("4. Canonical structures:", len(registry))
    print("5. GLB:", brain_glb)
    print("✓ COMPLETE")

    return manifest


# ============================================================
#  BATCH LOOP
# ============================================================

files = sorted(
    glob(os.path.join(IXI_DIR, "*.nii")) +
    glob(os.path.join(IXI_DIR, "*.nii.gz"))
)
print("\nFound", len(files), "MRI files")


results = []

for i, input_path in enumerate(files):

    filename = os.path.basename(input_path)

    # IXI014-HH-1236-T1.nii.gz → IXI014
    subject_id = filename.split("-")[0]

    print(
        f"\n[{i+1}/{len(files)}] {subject_id}"
    )

    try:

        manifest = process_subject(
            input_path,
            subject_id
        )

        results.append({
            "subject_id": subject_id,
            "status": "SUCCESS",
            "structures": len(
                manifest["structures"]
            )
        })

    except Exception as e:

        print(
            "ERROR:",
            repr(e)
        )

        results.append({
            "subject_id": subject_id,
            "status": "FAILED",
            "structures": 0
        })

        torch.cuda.empty_cache()
        gc.collect()


# ============================================================
# FINAL SUMMARY
# ============================================================

results_df = pd.DataFrame(results)

summary_path = os.path.join(
    BATCH_OUTPUT,
    "batch_summary.csv"
)

results_df.to_csv(
    summary_path,
    index=False
)

print("\n======================================")
print("BATCH COMPLETE")
print("======================================")
print(results_df["status"].value_counts())
print("Summary:", summary_path)