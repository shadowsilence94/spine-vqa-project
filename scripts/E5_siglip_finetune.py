# ============================================================
# E5 — SigLIP2 Fine-tuning Experiments (Fixed Version)
# ============================================================
# E5a: E4b + Unfreeze last 2 SigLIP2 layers
# E5b: E4b + Unfreeze last 4 SigLIP2 layers
# E5c: E4b + Full SigLIP2 fine-tuning
#
# Fixes applied (per ChatGPT review):
#   1. Patch tokens: no [:, 1:, :] slicing (SigLIP2 has no CLS)
#   2. Batch size: 16 (safer for fine-tuning)
#   3. SigLIP2: loaded in float32 (no .half() without AMP)
#   4. Image path: robust resolver
#   5. JSON: answer/answers fallback
#   6. Weight decay: 1e-4 (not 0.01)
#
# Different LRs per component:
#   SigLIP2 unfrozen: 1e-6 (very small!)
#   BERT:             2e-5
#   VertAttn + heads: 1e-4
# ============================================================

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor
from transformers import BertTokenizer, BertModel
from PIL import Image
from tqdm import tqdm
import numpy as np
import wandb
import argparse

# ============================================================
# 0. CONFIG
# ============================================================
class Config:
    DATA_ROOT     = "/home/dsia-st125985/SpineVQA/data/SpineBench"
    TRAIN_JSON    = f"{DATA_ROOT}/all/train_split.json"
    VAL_JSON      = f"{DATA_ROOT}/all/val_split.json"
    TEST_JSON     = f"{DATA_ROOT}/evaluation/test.json"
    IMG_ROOT      = f"{DATA_ROOT}/all"
    TEST_IMG_ROOT = f"{DATA_ROOT}/evaluation"
    SAVE_DIR      = "/home/dsia-st125985/SpineVQA/models"

    SIGLIP_NAME = "google/siglip2-base-patch16-224"
    BERT_NAME   = "bert-base-uncased"
    IMG_DIM     = 768
    Q_DIM       = 768
    HIDDEN_DIM  = 512
    NUM_DISEASE = 12
    NUM_LEVELS  = 5
    NUM_PATCHES = 196   # 14×14, no CLS in SigLIP2

    # Fix 2: batch size 16 for fine-tuning
    BATCH_SIZE  = 16
    EPOCHS      = 20
    DROPOUT     = 0.3
    MAX_LEN     = 64
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

    # Different LRs per component
    LR_SIGLIP   = 1e-6   # Very small for unfrozen SigLIP2
    LR_BERT     = 2e-5
    LR_NEW      = 1e-4   # VertAttn + heads

    # Fix 6: weight decay 1e-4
    WEIGHT_DECAY = 1e-4

    DISEASES = [
        "Subarticular Stenosis", "Foraminal stenosis", "Healthy",
        "Osteophytes", "Spinal Canal Stenosis", "cervical Lordosis",
        "Straight cervical vertebrae", "sigmoid cervical vertebrae",
        "cervical Kyphosis", "Disc space narrowing",
        "Spondylolisthesis", "Vertebral collapse"
    ]
    LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
    DISEASE2IDX = {d: i for i, d in enumerate(DISEASES)}
    LEVEL2IDX   = {l: i for i, l in enumerate(LEVELS)}

cfg = Config()
os.makedirs(cfg.SAVE_DIR, exist_ok=True)

import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# ============================================================
# 1. ROBUST IMAGE PATH RESOLVER (Fix 4)
# ============================================================
def resolve_image_path(img_root, sample):
    """Try multiple paths to find the image."""
    img = sample.get("image", sample.get("image_path", ""))
    base = os.path.basename(img)

    candidates = [
        os.path.join(img_root, img),
        os.path.join(img_root, base),
        os.path.join(cfg.DATA_ROOT, img),
        os.path.join(cfg.DATA_ROOT, base),
        os.path.join(cfg.DATA_ROOT, "all", img),
        os.path.join(cfg.DATA_ROOT, "all", base),
        os.path.join(cfg.DATA_ROOT, "evaluation", img),
        os.path.join(cfg.DATA_ROOT, "evaluation", base),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Image not found. Tried: {candidates[:4]}"
    )


# ============================================================
# 2. DATASET
# ============================================================
class SpineBenchDataset(Dataset):
    def __init__(self, json_path, img_root, processor,
                 tokenizer, split="train"):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.img_root  = img_root
        self.processor = processor
        self.tokenizer = tokenizer
        self.split     = split

        # Paired label injection
        self.image_disease  = {}
        self.image_location = {}
        for d in self.data:
            img = d["image"]
            if d["task"] == "spine_disease_classification":
                # Fix 5: answer/answers fallback
                ans = d.get("answers", d.get("answer", ""))
                if isinstance(ans, list): ans = ans[0]
                self.image_disease[img] = ans
            elif d["task"] == "spine_lesion_localization":
                ans = d.get("answers", d.get("answer", ""))
                self.image_location[img] = ans

        paired = set(self.image_disease) & set(self.image_location)
        print(f"Loaded {len(self.data):,} samples [{split}] | "
              f"paired: {len(paired):,}")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        sample   = self.data[idx]
        # Fix 4: robust image path
        img_path = resolve_image_path(self.img_root, sample)
        image    = Image.open(img_path).convert("RGB")
        img_tensor = self.processor(
            images=image, return_tensors="pt"
        )["pixel_values"].squeeze(0)

        tokens = self.tokenizer(
            sample.get("question", sample.get("query", "")),
            max_length=cfg.MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        input_ids      = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        task    = sample["task"]
        img_key = sample["image"]

        # Fix 5: answer/answers fallback
        raw_ans = sample.get("answers", sample.get("answer", ""))

        disease_label = -1
        if task == "spine_disease_classification":
            answer = raw_ans
            if isinstance(answer, list): answer = answer[0]
            disease_label = cfg.DISEASE2IDX.get(answer, -1)

        loc_label = torch.zeros(cfg.NUM_LEVELS)
        if task == "spine_lesion_localization":
            answers = raw_ans
            if isinstance(answers, str): answers = [answers]
            for ans in answers:
                if ans in cfg.LEVEL2IDX:
                    loc_label[cfg.LEVEL2IDX[ans]] = 1.0

        # Inject paired labels
        if task == "spine_disease_classification":
            if img_key in self.image_location:
                locs = self.image_location[img_key]
                if isinstance(locs, str): locs = [locs]
                for ans in locs:
                    if ans in cfg.LEVEL2IDX:
                        loc_label[cfg.LEVEL2IDX[ans]] = 1.0

        if task == "spine_lesion_localization":
            if img_key in self.image_disease:
                d_name = self.image_disease[img_key]
                disease_label = cfg.DISEASE2IDX.get(d_name, -1)

        return {
            "image":          img_tensor,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "task":           task,
            "disease_label":  torch.tensor(disease_label, dtype=torch.long),
            "loc_label":      loc_label,
        }


# ============================================================
# 3. MODEL
# ============================================================
class VertebralAttention(nn.Module):
    def __init__(self, dim=768, num_levels=5, num_heads=8):
        super().__init__()
        self.level_queries = nn.Parameter(torch.randn(num_levels, dim))
        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=0.1
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, patch_tokens):
        B       = patch_tokens.size(0)
        queries = self.level_queries.unsqueeze(0).expand(B, -1, -1)
        level_feats, attn_weights = self.attn(
            queries, patch_tokens, patch_tokens
        )
        return self.norm(level_feats), attn_weights


class E5Model(nn.Module):
    """
    E4b + configurable SigLIP2 unfreezing.
    unfreeze_layers:
        0  → Fully frozen (E4b)
        2  → E5a (last 2 layers)
        4  → E5b (last 4 layers)
        -1 → E5c (full fine-tune)
    """
    def __init__(self, unfreeze_layers=0):
        super().__init__()
        self.unfreeze_layers = unfreeze_layers

        # Fix 3: Load in float32, no torch_dtype=float16
        self.siglip2 = AutoModel.from_pretrained(cfg.SIGLIP_NAME)

        # Freeze all first
        for p in self.siglip2.parameters():
            p.requires_grad = False

        # Selectively unfreeze
        if unfreeze_layers == -1:
            for p in self.siglip2.parameters():
                p.requires_grad = True
            print("SigLIP2: FULLY UNFROZEN ✅")
        elif unfreeze_layers > 0:
            encoder_layers = self.siglip2.vision_model.encoder.layers
            total_layers   = len(encoder_layers)
            start_layer    = total_layers - unfreeze_layers
            for i, layer in enumerate(encoder_layers):
                if i >= start_layer:
                    for p in layer.parameters():
                        p.requires_grad = True
            for p in self.siglip2.vision_model.post_layernorm.parameters():
                p.requires_grad = True
            unfrozen = sum(
                p.numel() for p in self.siglip2.parameters()
                if p.requires_grad
            )
            print(f"SigLIP2: Last {unfreeze_layers} layers unfrozen "
                  f"({unfrozen:,} params) ✅")
        else:
            print("SigLIP2: FULLY FROZEN ❄️")

        self.bert           = BertModel.from_pretrained(cfg.BERT_NAME)
        self.vertebral_attn = VertebralAttention(cfg.IMG_DIM, cfg.NUM_LEVELS)

        self.image_proj = nn.Sequential(
            nn.Linear(cfg.IMG_DIM, cfg.HIDDEN_DIM), nn.ReLU()
        )
        self.question_proj = nn.Sequential(
            nn.Linear(cfg.Q_DIM, cfg.HIDDEN_DIM), nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM * 2, cfg.HIDDEN_DIM),
            nn.ReLU(), nn.Dropout(cfg.DROPOUT)
        )
        self.disease_head = nn.Linear(cfg.HIDDEN_DIM, cfg.NUM_DISEASE)
        self.loc_head     = nn.Linear(cfg.HIDDEN_DIM, cfg.NUM_LEVELS)

    def forward(self, images, input_ids, attention_mask):
        # Fix 1: No [:, 1:, :] — SigLIP2 has no CLS token
        # Fix 3: No images.half() — use float32
        if self.unfreeze_layers != 0:
            vision_out   = self.siglip2.vision_model(pixel_values=images)
            patch_tokens = vision_out.last_hidden_state  # [B, 196, 768]
        else:
            with torch.no_grad():
                vision_out   = self.siglip2.vision_model(pixel_values=images)
                patch_tokens = vision_out.last_hidden_state  # [B, 196, 768]

        # Fix 3: Patch token shape check
        if patch_tokens.size(1) != cfg.NUM_PATCHES:
            print(f"Warning: expected {cfg.NUM_PATCHES} patches, "
                  f"got {patch_tokens.size(1)}")

        level_feats, attn_weights = self.vertebral_attn(patch_tokens)
        f_img = level_feats.mean(dim=1)
        h_img = self.image_proj(f_img)

        q_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        f_q   = q_out.last_hidden_state[:, 0, :]
        h_q   = self.question_proj(f_q)

        fused          = torch.cat([h_img, h_q], dim=-1)
        z              = self.fusion(fused)
        disease_logits = self.disease_head(z)
        loc_logits     = self.loc_head(z)

        return disease_logits, loc_logits, z, attn_weights

    def get_param_groups(self):
        siglip_unfrozen, bert_params, new_params = [], [], []

        for p in self.siglip2.parameters():
            if p.requires_grad:
                siglip_unfrozen.append(p)

        for p in self.bert.parameters():
            bert_params.append(p)

        for m in [self.vertebral_attn, self.image_proj,
                  self.question_proj, self.fusion,
                  self.disease_head, self.loc_head]:
            for p in m.parameters():
                new_params.append(p)

        groups = []
        if siglip_unfrozen:
            groups.append({
                "params": siglip_unfrozen,
                "lr":     cfg.LR_SIGLIP,
                "name":   "siglip_unfrozen"
            })
        groups.append({"params": bert_params, "lr": cfg.LR_BERT,  "name": "bert"})
        groups.append({"params": new_params,  "lr": cfg.LR_NEW,   "name": "new_layers"})
        return groups


# ============================================================
# 4. LOSS
# ============================================================
def compute_class_weights(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    counts = torch.zeros(cfg.NUM_DISEASE)
    for s in data:
        if s["task"] == "spine_disease_classification":
            ans = s.get("answers", s.get("answer", ""))
            if isinstance(ans, list): ans = ans[0]
            idx = cfg.DISEASE2IDX.get(ans, -1)
            if idx >= 0: counts[idx] += 1
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * cfg.NUM_DISEASE
    print("\nClass weights:")
    for d, w in zip(cfg.DISEASES, weights):
        print(f"  {d[:30]:30s}: {w:.3f}")
    return weights


class TaskLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.disease_loss = nn.CrossEntropyLoss(
            weight=class_weights.to(cfg.DEVICE), ignore_index=-1
        )
        self.loc_loss = nn.BCEWithLogitsLoss()

    def forward(self, disease_logits, loc_logits,
                disease_labels, loc_labels, tasks):
        total_loss = torch.tensor(
            0.0, device=cfg.DEVICE, requires_grad=True
        )
        cls_mask = torch.tensor(
            [t == "spine_disease_classification" for t in tasks],
            dtype=torch.bool
        ).to(cfg.DEVICE)
        if cls_mask.any():
            total_loss = total_loss + self.disease_loss(
                disease_logits[cls_mask], disease_labels[cls_mask]
            )
        loc_mask = torch.tensor(
            [t == "spine_lesion_localization" for t in tasks],
            dtype=torch.bool
        ).to(cfg.DEVICE)
        if loc_mask.any():
            total_loss = total_loss + self.loc_loss(
                loc_logits[loc_mask], loc_labels[loc_mask]
            )
        return total_loss


# ============================================================
# 5. TRAIN + EVALUATE
# ============================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0; correct_cls = 0; total_cls = 0

    for batch in tqdm(loader, desc="Training"):
        images         = batch["image"].to(cfg.DEVICE)
        input_ids      = batch["input_ids"].to(cfg.DEVICE)
        attention_mask = batch["attention_mask"].to(cfg.DEVICE)
        disease_labels = batch["disease_label"].to(cfg.DEVICE)
        loc_labels     = batch["loc_label"].to(cfg.DEVICE)
        tasks          = batch["task"]

        optimizer.zero_grad()
        disease_logits, loc_logits, _, _ = model(
            images, input_ids, attention_mask
        )
        loss = criterion(
            disease_logits, loc_logits,
            disease_labels, loc_labels, tasks
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

        cls_mask = torch.tensor(
            [t == "spine_disease_classification" for t in tasks],
            dtype=torch.bool
        ).to(cfg.DEVICE)
        if cls_mask.any():
            preds = disease_logits[cls_mask].argmax(dim=1)
            correct_cls += (preds == disease_labels[cls_mask]).sum().item()
            total_cls   += cls_mask.sum().item()

    return (total_loss / len(loader),
            correct_cls / total_cls * 100 if total_cls > 0 else 0)


def evaluate(model, loader):
    model.eval()
    correct_cls = 0; total_cls   = 0
    exact_match = 0; total_loc   = 0
    all_preds   = []; all_gt     = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images         = batch["image"].to(cfg.DEVICE)
            input_ids      = batch["input_ids"].to(cfg.DEVICE)
            attention_mask = batch["attention_mask"].to(cfg.DEVICE)
            disease_labels = batch["disease_label"].to(cfg.DEVICE)
            loc_labels     = batch["loc_label"].to(cfg.DEVICE)
            tasks          = batch["task"]

            disease_logits, loc_logits, _, _ = model(
                images, input_ids, attention_mask
            )
            cls_mask = torch.tensor(
                [t == "spine_disease_classification" for t in tasks],
                dtype=torch.bool
            ).to(cfg.DEVICE)
            if cls_mask.any():
                preds = disease_logits[cls_mask].argmax(dim=1)
                correct_cls += (preds == disease_labels[cls_mask]).sum().item()
                total_cls   += cls_mask.sum().item()

            loc_mask = torch.tensor(
                [t == "spine_lesion_localization" for t in tasks],
                dtype=torch.bool
            ).to(cfg.DEVICE)
            if loc_mask.any():
                loc_preds   = (torch.sigmoid(loc_logits[loc_mask]) >= 0.5).float()
                loc_gt      = loc_labels[loc_mask]
                exact_match += (loc_preds == loc_gt).all(dim=1).sum().item()
                total_loc   += loc_mask.sum().item()
                all_preds.append(loc_preds.cpu())
                all_gt.append(loc_gt.cpu())

    cls_acc       = correct_cls / total_cls * 100 if total_cls > 0 else 0
    loc_exact_acc = exact_match / total_loc * 100 if total_loc > 0 else 0

    if all_preds:
        preds_cat = torch.cat(all_preds, dim=0)
        gt_cat    = torch.cat(all_gt,    dim=0)
        tp        = (preds_cat * gt_cat).sum(dim=1)
        precision = (tp / (preds_cat.sum(dim=1) + 1e-6)).mean().item() * 100
        recall    = (tp / (gt_cat.sum(dim=1) + 1e-6)).mean().item() * 100
    else:
        precision = recall = 0.0

    total_all   = total_cls + total_loc
    overall_acc = (correct_cls + exact_match) / total_all * 100 \
                  if total_all > 0 else 0

    return {
        "cls_acc":       cls_acc,
        "loc_exact_acc": loc_exact_acc,
        "precision":     precision,
        "recall":        recall,
        "overall_acc":   overall_acc,
    }


# ============================================================
# 6. MAIN
# ============================================================
def main(model_name):
    config_map = {
        "E5a": {"unfreeze": 2,  "desc": "Unfreeze last 2 SigLIP2 layers"},
        "E5b": {"unfreeze": 4,  "desc": "Unfreeze last 4 SigLIP2 layers"},
        "E5c": {"unfreeze": -1, "desc": "Full SigLIP2 fine-tuning"},
    }
    conf = config_map[model_name]

    print(f"\n{'='*65}")
    print(f"{model_name}: E4b + {conf['desc']}")
    print(f"Fixes: fp32, no CLS slice, robust paths, wd=1e-4")
    print(f"{'='*65}\n")

    wandb.init(
        project="SpineVQA-CL",
        name=f"{model_name}-{conf['desc'].replace(' ', '-')}",
        config={
            "model":            model_name,
            "description":      conf["desc"],
            "unfreeze_layers":  conf["unfreeze"],
            "lr_siglip":        cfg.LR_SIGLIP,
            "lr_bert":          cfg.LR_BERT,
            "lr_new":           cfg.LR_NEW,
            "weight_decay":     cfg.WEIGHT_DECAY,
            "batch_size":       cfg.BATCH_SIZE,
            "epochs":           cfg.EPOCHS,
            "split":            "image-level 80/20",
            "siglip_fp32":      True,
            "no_cls_slice":     True,
        }
    )

    processor = AutoImageProcessor.from_pretrained(cfg.SIGLIP_NAME)
    tokenizer = BertTokenizer.from_pretrained(cfg.BERT_NAME)

    train_ds = SpineBenchDataset(cfg.TRAIN_JSON, cfg.IMG_ROOT,      processor, tokenizer, "train")
    val_ds   = SpineBenchDataset(cfg.VAL_JSON,   cfg.IMG_ROOT,      processor, tokenizer, "val")
    test_ds  = SpineBenchDataset(cfg.TEST_JSON,  cfg.TEST_IMG_ROOT, processor, tokenizer, "test")

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = E5Model(unfreeze_layers=conf["unfreeze"]).to(cfg.DEVICE)

    total_p   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params:     {total_p:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Frozen params:    {total_p - trainable:,}")

    wandb.config.update({
        "total_params":     total_p,
        "trainable_params": trainable,
    })

    class_weights = compute_class_weights(cfg.TRAIN_JSON)
    criterion     = TaskLoss(class_weights)

    param_groups = model.get_param_groups()
    optimizer    = torch.optim.AdamW(
        param_groups, weight_decay=cfg.WEIGHT_DECAY  # Fix 6: 1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS
    )

    print(f"\nOptimizer groups:")
    for g in param_groups:
        n = sum(p.numel() for p in g["params"])
        print(f"  {g['name']:20s}: lr={g['lr']:.1e}, params={n:,}")

    best_val_overall = 0.0
    best_epoch       = 0

    print(f"\n{'='*90}")
    print(
        f"{'Epoch':>6} | {'Loss':>8} | {'TrainCls':>9} | "
        f"{'ValCls':>7} | {'ValLoc':>7} | {'ValPre':>7} | "
        f"{'ValRec':>7} | {'ValOverall':>11} | {'LR':>10}"
    )
    print("-" * 90)

    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion
        )
        val_metrics = evaluate(model, val_loader)
        scheduler.step()
        current_lr = optimizer.param_groups[-2]["lr"]

        wandb.log({
            "epoch":          epoch,
            "train_loss":     train_loss,
            "train_cls_acc":  train_acc,
            "val_cls_acc":    val_metrics["cls_acc"],
            "val_loc_acc":    val_metrics["loc_exact_acc"],
            "val_precision":  val_metrics["precision"],
            "val_recall":     val_metrics["recall"],
            "val_overall":    val_metrics["overall_acc"],
            "lr_bert":        current_lr,
            "lr_siglip":      optimizer.param_groups[0]["lr"]
                              if conf["unfreeze"] != 0 else 0,
        })

        if val_metrics["overall_acc"] > best_val_overall:
            best_val_overall = val_metrics["overall_acc"]
            best_epoch       = epoch
            torch.save(
                model.state_dict(),
                os.path.join(cfg.SAVE_DIR, f"{model_name}_best.pth")
            )
            saved = "✓"
        else:
            saved = ""

        print(
            f"{epoch:>6} | {train_loss:>8.4f} | {train_acc:>8.2f}% | "
            f"{val_metrics['cls_acc']:>6.2f}% | "
            f"{val_metrics['loc_exact_acc']:>6.2f}% | "
            f"{val_metrics['precision']:>6.2f}% | "
            f"{val_metrics['recall']:>6.2f}% | "
            f"{val_metrics['overall_acc']:>10.2f}% | "
            f"{current_lr:>10.2e}  {saved}"
        )

    print(f"\nBest Val Overall: {best_val_overall:.2f}% (Epoch {best_epoch})")

    # FINAL TEST
    print(f"\n{'='*65}")
    print("FINAL TEST EVALUATION (test.json — reported ONCE)")
    print(f"Loading best model from epoch {best_epoch}...")
    model.load_state_dict(
        torch.load(
            os.path.join(cfg.SAVE_DIR, f"{model_name}_best.pth"),
            map_location=cfg.DEVICE
        )
    )
    test_metrics = evaluate(model, test_loader)

    print(f"\n{model_name} FINAL TEST RESULTS:")
    print(f"  Disease Acc:  {test_metrics['cls_acc']:.2f}%")
    print(f"  Loc Acc:      {test_metrics['loc_exact_acc']:.2f}%")
    print(f"  Precision:    {test_metrics['precision']:.2f}%")
    print(f"  Recall:       {test_metrics['recall']:.2f}%")
    print(f"  Overall:      {test_metrics['overall_acc']:.2f}%")

    wandb.log({
        "test_cls_acc":      test_metrics["cls_acc"],
        "test_loc_acc":      test_metrics["loc_exact_acc"],
        "test_precision":    test_metrics["precision"],
        "test_recall":       test_metrics["recall"],
        "test_overall":      test_metrics["overall_acc"],
        "best_val_epoch":    best_epoch,
        "best_val_overall":  best_val_overall,
    })

    diff = test_metrics["overall_acc"] - 58.32
    print(f"\nComparison vs E4b (58.32%):")
    if diff > 0:
        print(f"  {model_name}: +{diff:.2f}% ✅ Improvement!")
    else:
        print(f"  {model_name}: {diff:.2f}% — E4b still best")

    wandb.finish()


# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["E5a", "E5b", "E5c"],
        help="E5a=last 2 layers, E5b=last 4 layers, E5c=full"
    )
    args = parser.parse_args()
    main(args.model)
