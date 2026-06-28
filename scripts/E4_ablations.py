# ============================================================
# E4 Ablations: Proving each component's contribution
# ============================================================
# E4a: Patch tokens + Mean Pool (no VertAttn, no QGFusion)
# E4b: Patch tokens + VertAttn + Simple Concat (no QGFusion)
# E4c: Full LAVP-Net (VertAttn + QGFusion) = E4
#
# Uses train_split.json for training
# Uses val_split.json for epoch selection
# Uses test.json for FINAL report only
# ============================================================

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
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
    TRAIN_JSON    = f"{DATA_ROOT}/all/train_split.json"   # 80%
    VAL_JSON      = f"{DATA_ROOT}/all/val_split.json"     # 20%
    TEST_JSON     = f"{DATA_ROOT}/evaluation/test.json"   # final
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
    NUM_PATCHES = 196

    BATCH_SIZE  = 32
    LR          = 2e-5
    EPOCHS      = 20
    DROPOUT     = 0.3
    MAX_LEN     = 64
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

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


# ============================================================
# 1. DATASET
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

        # Paired label maps
        self.image_disease  = {}
        self.image_location = {}
        for d in self.data:
            img = d["image"]
            if d["task"] == "spine_disease_classification":
                ans = d["answers"]
                if isinstance(ans, list): ans = ans[0]
                self.image_disease[img] = ans
            elif d["task"] == "spine_lesion_localization":
                self.image_location[img] = d["answers"]

        paired = set(self.image_disease) & set(self.image_location)
        print(f"Loaded {len(self.data):,} samples [{split}] | paired: {len(paired):,}")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        sample  = self.data[idx]
        img_path = os.path.join(self.img_root, sample["image"])
        image   = Image.open(img_path).convert("RGB")
        img_tensor = self.processor(
            images=image, return_tensors="pt"
        )["pixel_values"].squeeze(0)

        tokens = self.tokenizer(
            sample["question"], max_length=cfg.MAX_LEN,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids      = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        task    = sample["task"]
        img_key = sample["image"]

        disease_label = -1
        if task == "spine_disease_classification":
            answer = sample["answers"]
            if isinstance(answer, list): answer = answer[0]
            disease_label = cfg.DISEASE2IDX.get(answer, -1)

        loc_label = torch.zeros(cfg.NUM_LEVELS)
        if task == "spine_lesion_localization":
            answers = sample["answers"]
            if isinstance(answers, str): answers = [answers]
            for ans in answers:
                if ans in cfg.LEVEL2IDX:
                    loc_label[cfg.LEVEL2IDX[ans]] = 1.0

        # Inject paired labels
        if task == "spine_disease_classification":
            if img_key in self.image_location:
                answers = self.image_location[img_key]
                if isinstance(answers, str): answers = [answers]
                for ans in answers:
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
# 2. MODELS
# ============================================================

# ── 2a. E4a: Patch Mean Pool (no VertAttn, no QGFusion) ────
class E4a_PatchMeanPool(nn.Module):
    """
    E4a Ablation: Switch CLS to mean of patch tokens.
    No VertebralAttention, no QuestionGuidedFusion.
    Measures: how much does using patch tokens help?
    """
    def __init__(self):
        super().__init__()
        self.siglip2 = AutoModel.from_pretrained(
            cfg.SIGLIP_NAME, torch_dtype=torch.float16
        )
        for p in self.siglip2.parameters():
            p.requires_grad = False

        self.bert = BertModel.from_pretrained(cfg.BERT_NAME)

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
        with torch.no_grad():
            vision_out = self.siglip2.vision_model(
                pixel_values=images.half()
            )
            all_tokens   = vision_out.last_hidden_state.float()
            patch_tokens = all_tokens[:, 1:, :]  # [B, 196, 768]

        # Mean pool over all patches (no attention)
        f_img = patch_tokens.mean(dim=1)           # [B, 768]
        h_img = self.image_proj(f_img)

        q_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        f_q   = q_out.last_hidden_state[:, 0, :]
        h_q   = self.question_proj(f_q)

        fused          = torch.cat([h_img, h_q], dim=-1)
        z              = self.fusion(fused)
        disease_logits = self.disease_head(z)
        loc_logits     = self.loc_head(z)
        return disease_logits, loc_logits, z


# ── 2b. E4b: VertAttn + Simple Concat (no QGFusion) ────────
class VertebralAttention(nn.Module):
    def __init__(self, dim=768, num_levels=5, num_heads=8):
        super().__init__()
        self.level_queries = nn.Parameter(torch.randn(num_levels, dim))
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.1)
        self.norm  = nn.LayerNorm(dim)

    def forward(self, patch_tokens):
        B       = patch_tokens.size(0)
        queries = self.level_queries.unsqueeze(0).expand(B, -1, -1)
        level_feats, attn_weights = self.attn(queries, patch_tokens, patch_tokens)
        return self.norm(level_feats), attn_weights


class E4b_VertAttnSimple(nn.Module):
    """
    E4b Ablation: VertebralAttention + simple mean of level features.
    No QuestionGuidedFusion.
    Measures: how much does VertAttn help vs mean pool?
    """
    def __init__(self):
        super().__init__()
        self.siglip2 = AutoModel.from_pretrained(
            cfg.SIGLIP_NAME, torch_dtype=torch.float16
        )
        for p in self.siglip2.parameters():
            p.requires_grad = False

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
        with torch.no_grad():
            vision_out   = self.siglip2.vision_model(pixel_values=images.half())
            patch_tokens = vision_out.last_hidden_state.float()[:, 1:, :]

        # VertAttn → 5 level features
        level_feats, attn_weights = self.vertebral_attn(patch_tokens)
        # Simple mean of 5 levels (no question guidance)
        f_img = level_feats.mean(dim=1)            # [B, 768]
        h_img = self.image_proj(f_img)

        q_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        f_q   = q_out.last_hidden_state[:, 0, :]
        h_q   = self.question_proj(f_q)

        fused          = torch.cat([h_img, h_q], dim=-1)
        z              = self.fusion(fused)
        disease_logits = self.disease_head(z)
        loc_logits     = self.loc_head(z)
        return disease_logits, loc_logits, z, attn_weights


# ── 2c. E4c: Full LAVP-Net (VertAttn + QGFusion) ────────────
class QuestionGuidedFusion(nn.Module):
    def __init__(self, img_dim=768, q_dim=768, out_dim=512, num_heads=8):
        super().__init__()
        self.q_proj     = nn.Linear(q_dim, img_dim)
        self.cross_attn = nn.MultiheadAttention(img_dim, num_heads, batch_first=True, dropout=0.1)
        self.norm1      = nn.LayerNorm(img_dim)
        self.fusion_proj = nn.Sequential(
            nn.Linear(img_dim + q_dim, out_dim), nn.ReLU(), nn.Dropout(0.3)
        )

    def forward(self, level_feats, q_feat):
        q_query  = self.q_proj(q_feat).unsqueeze(1)
        attn_out, _ = self.cross_attn(q_query, level_feats, level_feats)
        attn_out = self.norm1(attn_out).squeeze(1)
        fused    = torch.cat([attn_out, q_feat], dim=-1)
        return self.fusion_proj(fused)


class E4c_FullLAVPNet(nn.Module):
    """
    E4c: Full LAVP-Net = VertAttn + QGFusion (same as E4).
    Included here for fair comparison with same train/val split.
    """
    def __init__(self):
        super().__init__()
        self.siglip2 = AutoModel.from_pretrained(
            cfg.SIGLIP_NAME, torch_dtype=torch.float16
        )
        for p in self.siglip2.parameters():
            p.requires_grad = False

        self.bert           = BertModel.from_pretrained(cfg.BERT_NAME)
        self.vertebral_attn = VertebralAttention(cfg.IMG_DIM, cfg.NUM_LEVELS)
        self.qg_fusion      = QuestionGuidedFusion(cfg.IMG_DIM, cfg.Q_DIM, cfg.HIDDEN_DIM)
        self.disease_head   = nn.Linear(cfg.HIDDEN_DIM, cfg.NUM_DISEASE)
        self.loc_head       = nn.Linear(cfg.HIDDEN_DIM, cfg.NUM_LEVELS)

    def forward(self, images, input_ids, attention_mask):
        with torch.no_grad():
            vision_out   = self.siglip2.vision_model(pixel_values=images.half())
            patch_tokens = vision_out.last_hidden_state.float()[:, 1:, :]

        level_feats, attn_weights = self.vertebral_attn(patch_tokens)

        q_out  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        q_feat = q_out.last_hidden_state[:, 0, :]

        z              = self.qg_fusion(level_feats, q_feat)
        disease_logits = self.disease_head(z)
        loc_logits     = self.loc_head(z)
        return disease_logits, loc_logits, z, attn_weights


# ============================================================
# 3. LOSS
# ============================================================
def compute_class_weights(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    counts = torch.zeros(cfg.NUM_DISEASE)
    for s in data:
        if s["task"] == "spine_disease_classification":
            ans = s["answers"]
            if isinstance(ans, list): ans = ans[0]
            idx = cfg.DISEASE2IDX.get(ans, -1)
            if idx >= 0: counts[idx] += 1
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * cfg.NUM_DISEASE
    print("Class weights:")
    for i, (d, w) in enumerate(zip(cfg.DISEASES, weights)):
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
        total_loss = torch.tensor(0.0, device=cfg.DEVICE, requires_grad=True)

        cls_mask = torch.tensor(
            [t == "spine_disease_classification" for t in tasks],
            dtype=torch.bool
        ).to(cfg.DEVICE)
        if cls_mask.any():
            L_d = self.disease_loss(disease_logits[cls_mask], disease_labels[cls_mask])
            total_loss = total_loss + L_d

        loc_mask = torch.tensor(
            [t == "spine_lesion_localization" for t in tasks],
            dtype=torch.bool
        ).to(cfg.DEVICE)
        if loc_mask.any():
            L_l = self.loc_loss(loc_logits[loc_mask], loc_labels[loc_mask])
            total_loss = total_loss + L_l

        return total_loss


# ============================================================
# 4. TRAIN + EVALUATE
# ============================================================
def train_epoch(model, loader, optimizer, criterion, model_type):
    model.train()
    total_loss = 0.0
    correct_cls = 0
    total_cls   = 0

    for batch in tqdm(loader, desc="Training"):
        images         = batch["image"].to(cfg.DEVICE)
        input_ids      = batch["input_ids"].to(cfg.DEVICE)
        attention_mask = batch["attention_mask"].to(cfg.DEVICE)
        disease_labels = batch["disease_label"].to(cfg.DEVICE)
        loc_labels     = batch["loc_label"].to(cfg.DEVICE)
        tasks          = batch["task"]

        optimizer.zero_grad()

        if model_type == "E4a":
            disease_logits, loc_logits, _ = model(images, input_ids, attention_mask)
        else:
            disease_logits, loc_logits, _, _ = model(images, input_ids, attention_mask)

        loss = criterion(disease_logits, loc_logits, disease_labels, loc_labels, tasks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

        cls_mask = torch.tensor(
            [t == "spine_disease_classification" for t in tasks], dtype=torch.bool
        ).to(cfg.DEVICE)
        if cls_mask.any():
            preds = disease_logits[cls_mask].argmax(dim=1)
            correct_cls += (preds == disease_labels[cls_mask]).sum().item()
            total_cls   += cls_mask.sum().item()

    return total_loss / len(loader), correct_cls / total_cls * 100 if total_cls > 0 else 0


def evaluate(model, loader, model_type):
    model.eval()
    correct_cls = 0; total_cls = 0
    exact_match = 0; total_loc = 0
    all_preds = []; all_gt = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images         = batch["image"].to(cfg.DEVICE)
            input_ids      = batch["input_ids"].to(cfg.DEVICE)
            attention_mask = batch["attention_mask"].to(cfg.DEVICE)
            disease_labels = batch["disease_label"].to(cfg.DEVICE)
            loc_labels     = batch["loc_label"].to(cfg.DEVICE)
            tasks          = batch["task"]

            if model_type == "E4a":
                disease_logits, loc_logits, _ = model(images, input_ids, attention_mask)
            else:
                disease_logits, loc_logits, _, _ = model(images, input_ids, attention_mask)

            cls_mask = torch.tensor(
                [t == "spine_disease_classification" for t in tasks], dtype=torch.bool
            ).to(cfg.DEVICE)
            if cls_mask.any():
                preds = disease_logits[cls_mask].argmax(dim=1)
                correct_cls += (preds == disease_labels[cls_mask]).sum().item()
                total_cls   += cls_mask.sum().item()

            loc_mask = torch.tensor(
                [t == "spine_lesion_localization" for t in tasks], dtype=torch.bool
            ).to(cfg.DEVICE)
            if loc_mask.any():
                loc_preds = (torch.sigmoid(loc_logits[loc_mask]) >= 0.5).float()
                loc_gt    = loc_labels[loc_mask]
                exact     = (loc_preds == loc_gt).all(dim=1).sum().item()
                exact_match += exact
                total_loc   += loc_mask.sum().item()
                all_preds.append(loc_preds.cpu())
                all_gt.append(loc_gt.cpu())

    cls_acc       = correct_cls / total_cls * 100 if total_cls > 0 else 0
    loc_exact_acc = exact_match / total_loc * 100 if total_loc > 0 else 0

    if all_preds:
        preds_cat = torch.cat(all_preds, dim=0)
        gt_cat    = torch.cat(all_gt, dim=0)
        tp        = (preds_cat * gt_cat).sum(dim=1)
        precision = (tp / (preds_cat.sum(dim=1) + 1e-6)).mean().item() * 100
        recall    = (tp / (gt_cat.sum(dim=1) + 1e-6)).mean().item() * 100
    else:
        precision = recall = 0.0

    total_all   = total_cls + total_loc
    overall_acc = (correct_cls + exact_match) / total_all * 100 if total_all > 0 else 0

    return {"cls_acc": cls_acc, "loc_exact_acc": loc_exact_acc,
            "precision": precision, "recall": recall, "overall_acc": overall_acc}


# ============================================================
# 5. MAIN
# ============================================================
def main(model_name):
    print(f"\n{'='*65}")
    print(f"E4 Ablation: {model_name}")
    print(f"Train/Val/Test split (image-level, no leakage)")
    print(f"{'='*65}\n")

    wandb.init(
        project="SpineVQA-CL",
        name=f"Ablation-{model_name}",
        config={"model": model_name, "lr": cfg.LR,
                "batch_size": cfg.BATCH_SIZE, "epochs": cfg.EPOCHS,
                "split": "image-level 80/20"}
    )

    from transformers import SiglipImageProcessor
    processor = SiglipImageProcessor.from_pretrained(cfg.SIGLIP_NAME)
    tokenizer = BertTokenizer.from_pretrained(cfg.BERT_NAME)

    train_ds = SpineBenchDataset(cfg.TRAIN_JSON, cfg.IMG_ROOT, processor, tokenizer, "train")
    val_ds   = SpineBenchDataset(cfg.VAL_JSON,   cfg.IMG_ROOT, processor, tokenizer, "val")
    test_ds  = SpineBenchDataset(cfg.TEST_JSON,  cfg.TEST_IMG_ROOT, processor, tokenizer, "test")

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Select model
    if model_name == "E4a":
        model = E4a_PatchMeanPool().to(cfg.DEVICE)
    elif model_name == "E4b":
        model = E4b_VertAttnSimple().to(cfg.DEVICE)
    elif model_name == "E4c":
        model = E4c_FullLAVPNet().to(cfg.DEVICE)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    total_p    = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_p:,}")
    print(f"Trainable params: {trainable:,}")

    class_weights = compute_class_weights(cfg.TRAIN_JSON)
    criterion     = TaskLoss(class_weights)
    optimizer     = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LR, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    # Training — use VAL for epoch selection
    best_val_overall = 0.0
    best_epoch       = 0
    print(f"\n{'='*80}")
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Cls':>8} | {'Loc':>8} | "
          f"{'Pre':>7} | {'Rec':>7} | {'Val Overall':>11} | {'LR':>10}")
    print("-" * 80)

    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, model_name)
        val_metrics  = evaluate(model, val_loader, model_name)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        wandb.log({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_cls_acc": val_metrics["cls_acc"],
            "val_loc_acc": val_metrics["loc_exact_acc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_overall": val_metrics["overall_acc"],
            "lr": current_lr,
        })

        # Save best based on VAL
        if val_metrics["overall_acc"] > best_val_overall:
            best_val_overall = val_metrics["overall_acc"]
            best_epoch       = epoch
            torch.save(model.state_dict(),
                       os.path.join(cfg.SAVE_DIR, f"{model_name}_best.pth"))
            saved = "✓"
        else:
            saved = ""

        print(f"{epoch:>6} | {train_loss:>8.4f} | {train_acc:>7.2f}% | "
              f"{val_metrics['loc_exact_acc']:>7.2f}% | "
              f"{val_metrics['precision']:>6.2f}% | "
              f"{val_metrics['recall']:>6.2f}% | "
              f"{val_metrics['overall_acc']:>10.2f}% | "
              f"{current_lr:>10.2e}  {saved}")

    print(f"\nBest Val Overall: {best_val_overall:.2f}% (Epoch {best_epoch})")

    # ── FINAL TEST EVALUATION (once!) ─────────────────────
    print(f"\n{'='*65}")
    print(f"FINAL TEST EVALUATION (test.json — reported once)")
    print(f"Loading best model from epoch {best_epoch}...")
    print(f"{'='*65}")

    model.load_state_dict(torch.load(
        os.path.join(cfg.SAVE_DIR, f"{model_name}_best.pth")
    ))
    test_metrics = evaluate(model, test_loader, model_name)

    print(f"\n{model_name} FINAL TEST RESULTS:")
    print(f"  Disease Acc:  {test_metrics['cls_acc']:.2f}%")
    print(f"  Loc Acc:      {test_metrics['loc_exact_acc']:.2f}%")
    print(f"  Precision:    {test_metrics['precision']:.2f}%")
    print(f"  Recall:       {test_metrics['recall']:.2f}%")
    print(f"  Overall:      {test_metrics['overall_acc']:.2f}%")

    wandb.log({
        "test_cls_acc":   test_metrics["cls_acc"],
        "test_loc_acc":   test_metrics["loc_exact_acc"],
        "test_precision": test_metrics["precision"],
        "test_recall":    test_metrics["recall"],
        "test_overall":   test_metrics["overall_acc"],
        "best_val_epoch": best_epoch,
    })

    wandb.finish()
    print(f"\nModel saved: {cfg.SAVE_DIR}/{model_name}_best.pth")


# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["E4a", "E4b", "E4c"],
                        help="Which ablation to run")
    args = parser.parse_args()
    main(args.model)
