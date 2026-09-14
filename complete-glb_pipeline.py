# ============================================================
# COMPLETE SINGLE-SUBJECT PIPELINE
# T1 → RAS → MNI305 → UNesT → Canonical Ontology → Meshes → GLB
# ============================================================

import os, json, shutil
import numpy as np
import pandas as pd
import nibabel as nib
import ants
import torch
import trimesh

from scipy import ndimage
from skimage.measure import marching_cubes
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    NormalizeIntensityd, EnsureTyped
)
from monai.inferers import SlidingWindowInferer


# -----------------------------
# PATHS
# -----------------------------
INPUT_MRI = "/kaggle/input/datasets/jackontreesandminds/ixi-t1-basemri/IXI325-Guys-0911-T1.nii/IXI325-Guys-0911-MPRAGESEN_-s032_-0301-00003-000001-01.nii"
MNI305_TEMPLATE = "/kaggle/input/datasets/jackontreesandminds/mni305-template/average305_t1_tal_lin.nii"
ONTOLOGY_CSV = "/kaggle/input/datasets/jackontreesandminds/required-files/canonical_brain_ontology_v2_FINAL.csv"
MAPPING_CSV = "/kaggle/input/datasets/jackontreesandminds/required-files/source_mapping_MONAI_UNEST_v0.2.7_FINAL.csv"

SUBJECT_ID = "IXI325"

OUTPUT_DIR = f"/kaggle/working/{SUBJECT_ID}_UNEST"
RAS_DIR = os.path.join(OUTPUT_DIR, "RAS")
MNI_DIR = os.path.join(OUTPUT_DIR, "MNI305")
TRANSFORM_DIR = os.path.join(MNI_DIR, "transforms")
MESH_DIR = os.path.join(MNI_DIR, "meshes")

for d in [RAS_DIR, MNI_DIR, TRANSFORM_DIR, MESH_DIR]:
    os.makedirs(d, exist_ok=True)


# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

print("Device:", device)


# ============================================================
# 1. RAS REORIENTATION
# ============================================================

def to_ras(input_path, output_path):
    img = nib.load(input_path)

    # Correct RAS reorientation with affine preservation
    ras_img = nib.as_closest_canonical(img)

    nib.save(ras_img, output_path)

    print("RAS:", ras_img.shape, nib.aff2axcodes(ras_img.affine))
    return output_path


ras_path = os.path.join(
    RAS_DIR,
    f"{SUBJECT_ID}_T1_RAS.nii.gz"
)

to_ras(INPUT_MRI, ras_path)


# ============================================================
# 2. REGISTER RAS → MNI305
# ============================================================

fixed = ants.image_read(MNI305_TEMPLATE)
moving = ants.image_read(ras_path)

registration = ants.registration(
    fixed=fixed,
    moving=moving,
    type_of_transform="Affine"
)

registered_path = os.path.join(
    MNI_DIR,
    f"{SUBJECT_ID}_T1_MNI305.nii.gz"
)

ants.image_write(
    registration["warpedmovout"],
    registered_path
)

# Preserve forward transform
forward_transform = registration["fwdtransforms"][0]

transform_path = os.path.join(
    TRANSFORM_DIR,
    f"{SUBJECT_ID}_to_MNI305_Affine.mat"
)

shutil.copy2(forward_transform, transform_path)

print("Registered:", registered_path)
print("Transform:", transform_path)


# ============================================================
# 3. MONAI PREPROCESSING
# ============================================================

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

data = preprocess({"image": registered_path})

image = data["image"].unsqueeze(0).to(device)

print("Input tensor:", image.shape)


# ============================================================
# 4. UNEST INFERENCE
# ============================================================

inferer = SlidingWindowInferer(
    roi_size=(96, 96, 96),
    sw_batch_size=1,
    overlap=0.7
)

with torch.inference_mode():
    logits = inferer(image, model)

labels = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

print("Segmentation:", labels.shape)
print("Labels:", len(np.unique(labels)))


# ============================================================
# 5. SAVE SEGMENTATION
# ============================================================

reference = nib.load(registered_path)

segmentation_path = os.path.join(
    MNI_DIR,
    f"{SUBJECT_ID}_UNEST_segmentation_MNI305.nii.gz"
)

seg_img = nib.Nifti1Image(
    labels,
    reference.affine,
    reference.header.copy()
)

seg_img.set_data_dtype(np.uint8)
nib.save(seg_img, segmentation_path)

print("Segmentation saved:", segmentation_path)


# ============================================================
# 6. LOAD ONTOLOGY + SOURCE MAPPING
# ============================================================

ontology = pd.read_csv(ONTOLOGY_CSV)
mapping = pd.read_csv(MAPPING_CSV)

mapping["source_id"] = mapping["source_id"].astype(int)

# ontology lookup
ontology_lookup = ontology.set_index("canonical_id").to_dict("index")

# source lookup
mapping_lookup = mapping.set_index("source_id").to_dict("index")

print("Ontology concepts:", len(ontology))
print("Mappings:", len(mapping))


# ============================================================
# 7. CREATE MESHES
# ============================================================

mesh_registry = []

affine = reference.affine

for source_id in sorted(mapping_lookup):

    if source_id == 0:
        continue

    info = mapping_lookup[source_id]

    canonical_id = info["canonical_id"]
    laterality = info["laterality"]
    source_name = info["source_name"]

    mask = labels == source_id

    voxel_count = int(mask.sum())

    if voxel_count == 0:
        continue

    # Keep largest connected component
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

    # voxel coordinates → MNI world coordinates
    verts_world = nib.affines.apply_affine(
        affine,
        verts
    )

    mesh = trimesh.Trimesh(
        vertices=verts_world,
        faces=faces,
        process=True
    )

    # Remove unwanted geometry
    _, unique_indices = trimesh.grouping.unique_rows(mesh.faces)
    mesh.update_faces(unique_indices)

    if mesh.volume < 0:
        mesh.invert()

    object_name = (
        f"{canonical_id}_{laterality}_{source_id:03d}"
    )

    mesh_path = os.path.join(
        MESH_DIR,
        object_name + ".glb"
    )

    mesh.export(mesh_path)

    concept = ontology_lookup.get(canonical_id, {})

    mesh_registry.append({
        "object_name": object_name,
        "source": info.get("source", "MONAI_UNesT"),
        "source_version": info.get("source_version", "0.2.7"),
        "source_id": source_id,
        "source_name": source_name,
        "canonical_id": canonical_id,
        "preferred_name": concept.get(
            "preferred_name",
            source_name
        ),
        "parent_id": concept.get("parent_id"),
        "category": concept.get("category"),
        "ontology_level": concept.get("ontology_level"),
        "laterality": laterality,
        "mesh_file": os.path.basename(mesh_path),
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "volume_mm3": abs(float(mesh.volume)),
        "watertight": bool(mesh.is_watertight)
    })


mesh_registry = pd.DataFrame(mesh_registry)

registry_path = os.path.join(
    MNI_DIR,
    "mesh_registry.csv"
)

mesh_registry.to_csv(
    registry_path,
    index=False
)

print("Meshes:", len(mesh_registry))


# ============================================================
# 8. COMBINE ALL MESHES → brain.glb
# ============================================================

scene = trimesh.Scene()

for _, row in mesh_registry.iterrows():

    mesh_path = os.path.join(
        MESH_DIR,
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
    MNI_DIR,
    "brain.glb"
)

scene.export(brain_glb)

print("Brain GLB:", brain_glb)


# ============================================================
# 9. MANIFEST
# ============================================================

manifest = {
    "schema": "1.0",
    "subject_id": SUBJECT_ID,

    "model": {
        "name": "MONAI Whole Brain Large UNesT",
        "version": "0.2.7",
        "source": "MONAI",
        "label_count": 133
    },

    "space": {
        "name": "MNI305",
        "coordinate_system": "RAS",
        "voxel_spacing_mm": [1.0, 1.0, 1.0]
    },

    "ontology": {
        "name": "Canonical Brain Ontology",
        "version": "2.0"
    },

    "segmentation": os.path.basename(segmentation_path),

    "registration": {
        "transform": os.path.relpath(
            transform_path,
            MNI_DIR
        )
    },

    "structures": mesh_registry.to_dict(
        orient="records"
    )
}

manifest_path = os.path.join(
    MNI_DIR,
    "manifest.json"
)

with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)


print("\n================================")
print("PIPELINE COMPLETE")
print("================================")
print("Subject:", SUBJECT_ID)
print("Structures:", len(mesh_registry))
print("Segmentation:", segmentation_path)
print("GLB:", brain_glb)
print("Manifest:", manifest_path)