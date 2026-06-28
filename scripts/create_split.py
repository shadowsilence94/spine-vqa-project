"""
SpineBench Image-Level Stratified Train/Val Split
==================================================
CORRECT approach: split by IMAGE ID, not sample ID.

Same image may have both:
  - classification sample
  - localization sample

So we must ensure all samples from same image
go to the SAME split (train or val).

Logic:
  1. Group all samples by image ID
  2. Assign stratification label per image
     (use disease label if available, else level)
  3. Split IMAGE IDs into train/val (stratified)
  4. Collect all samples for train images
  5. Collect all samples for val images
  6. Verify ZERO image overlap

Usage:
  python3 create_split.py
"""

import json
import random
import os
from collections import defaultdict, Counter

# ── Config ────────────────────────────────────────────────
TRAIN_JSON  = "/home/dsia-st125985/SpineVQA/data/SpineBench/all/train.json"
OUTPUT_DIR  = "/home/dsia-st125985/SpineVQA/data/SpineBench/all"
VAL_RATIO   = 0.20
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ── Load ──────────────────────────────────────────────────
print("Loading train.json...")
with open(TRAIN_JSON, "r") as f:
    data = json.load(f)
print(f"Total samples: {len(data):,}")

# ── Step 1: Group samples by image ────────────────────────
image_to_samples = defaultdict(list)
for s in data:
    image_to_samples[s["image"]].append(s)

print(f"Unique images: {len(image_to_samples):,}")

# ── Step 2: Assign stratification label per image ─────────
# Priority: use disease label if image has classification sample
# Otherwise: use localization level
image_disease  = {}
image_location = {}

for s in data:
    img = s["image"]
    if s["task"] == "spine_disease_classification":
        ans = s["answers"]
        if isinstance(ans, list): ans = ans[0]
        image_disease[img] = ans
    elif s["task"] == "spine_lesion_localization":
        ans = s["answers"]
        if isinstance(ans, list): ans = ans[0]
        image_location[img] = ans

# Assign one stratification label per image
image_strat = {}
for img in image_to_samples:
    if img in image_disease:
        image_strat[img] = ("disease", image_disease[img])
    else:
        image_strat[img] = ("level", image_location.get(img, "unknown"))

# Count per stratum
strat_groups = defaultdict(list)
for img, label in image_strat.items():
    strat_groups[label].append(img)

print(f"\nStratification groups:")
for label, imgs in sorted(strat_groups.items(), key=lambda x: -len(x[1])):
    print(f"  {str(label):45s}: {len(imgs):5d} images")

# ── Step 3: Split IMAGE IDs (stratified) ──────────────────
train_images = set()
val_images   = set()

print(f"\nImage-level split:")
for label, imgs in sorted(strat_groups.items()):
    random.shuffle(imgs)
    n_val   = max(1, int(len(imgs) * VAL_RATIO))
    n_train = len(imgs) - n_val
    train_images.update(imgs[n_val:])
    val_images.update(imgs[:n_val])
    print(f"  {str(label):45s}: {len(imgs):5d} → train={n_train}, val={n_val}")

# ── Step 4 & 5: Collect samples ───────────────────────────
train_split = []
val_split   = []

for s in data:
    if s["image"] in train_images:
        train_split.append(s)
    elif s["image"] in val_images:
        val_split.append(s)

random.shuffle(train_split)
random.shuffle(val_split)

# ── Step 6: Verify ZERO image overlap ─────────────────────
print(f"\n{'='*55}")
print("OVERLAP CHECK:")
train_imgs_check = set(d["image"] for d in train_split)
val_imgs_check   = set(d["image"] for d in val_split)
overlap = train_imgs_check & val_imgs_check
print(f"  Train images: {len(train_imgs_check):,}")
print(f"  Val images:   {len(val_imgs_check):,}")
print(f"  Overlap:      {len(overlap)}")
assert len(overlap) == 0, f"LEAKAGE DETECTED! {len(overlap)} images overlap!"
print("  STATUS: NO LEAKAGE ✅")

# ── Summary ───────────────────────────────────────────────
print(f"\n{'='*55}")
print("FINAL SPLIT SUMMARY")
print(f"{'='*55}")
print(f"Total original samples: {len(data):,}")
print(f"Train samples:          {len(train_split):,} ({len(train_split)/len(data)*100:.1f}%)")
print(f"Val samples:            {len(val_split):,}  ({len(val_split)/len(data)*100:.1f}%)")
print(f"Total images:           {len(image_to_samples):,}")
print(f"Train images:           {len(train_imgs_check):,}")
print(f"Val images:             {len(val_imgs_check):,}")

# Task breakdown
print(f"\nTrain task breakdown:")
train_tasks = Counter(d["task"] for d in train_split)
print(f"  Classification: {train_tasks['spine_disease_classification']:,}")
print(f"  Localization:   {train_tasks['spine_lesion_localization']:,}")

print(f"\nVal task breakdown:")
val_tasks = Counter(d["task"] for d in val_split)
print(f"  Classification: {val_tasks['spine_disease_classification']:,}")
print(f"  Localization:   {val_tasks['spine_lesion_localization']:,}")

# Disease distribution
print(f"\nDisease class distribution:")
train_cls = [d for d in train_split if d["task"]=="spine_disease_classification"]
val_cls   = [d for d in val_split   if d["task"]=="spine_disease_classification"]
train_dis = Counter(d["answers"][0] if isinstance(d["answers"],list)
                    else d["answers"] for d in train_cls)
val_dis   = Counter(d["answers"][0] if isinstance(d["answers"],list)
                    else d["answers"] for d in val_cls)

DISEASES = [
    "Subarticular Stenosis","Foraminal stenosis","Healthy",
    "Osteophytes","Spinal Canal Stenosis","cervical Lordosis",
    "Straight cervical vertebrae","sigmoid cervical vertebrae",
    "cervical Kyphosis","Disc space narrowing",
    "Spondylolisthesis","Vertebral collapse"
]

print(f"  {'Disease':33s} {'Train':>6} {'Val':>5} {'Val%':>6}")
print("  " + "-"*53)
for dis in DISEASES:
    t = train_dis.get(dis, 0)
    v = val_dis.get(dis, 0)
    pct = v/(t+v)*100 if (t+v)>0 else 0
    print(f"  {dis[:33]:33s} {t:>6} {v:>5} {pct:>5.1f}%")

# ── Save ──────────────────────────────────────────────────
train_path = os.path.join(OUTPUT_DIR, "train_split.json")
val_path   = os.path.join(OUTPUT_DIR, "val_split.json")

with open(train_path, "w") as f:
    json.dump(train_split, f)
with open(val_path, "w") as f:
    json.dump(val_split, f)

print(f"\n{'='*55}")
print("Files saved:")
print(f"  {train_path}")
print(f"  {val_path}")
print(f"\nTest set (untouched):")
print(f"  .../evaluation/test.json (2,128 samples)")
print(f"{'='*55}")
print("Done!")
