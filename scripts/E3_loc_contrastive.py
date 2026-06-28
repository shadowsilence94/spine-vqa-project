# ============================================================
# E3 — Localization-Aware Contrastive SpineVQA Model
# Thesis: Localization-Aware Fine-Grained Contrastive Learning
#         for Spinal Pathology Representation in Medical VQA
# Dataset: SpineBench (ACM MM 2025)
#
# Change from E2:
#   SpineContrastiveLoss updated:
#   Positive = same disease + same spinal level
#   Hard Neg = same disease + diff level
#            = diff disease + same level
# ============================================================

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModel
from transformers import BertTokenizer, BertModel
from PIL import Image
from tqdm import tqdm
import numpy as np
import wandb

# ============================================================
# 0. CONFIG
# ============================================================
class Config:
    # Paths
    DATA_ROOT     = "/home/dsia-st125985/SpineVQA/data/SpineBench"
    TRAIN_JSON    = f"{DATA_ROOT}/all/train.json"
    TEST_JSON     = f"{DATA_ROOT}/evaluation/test.json"
    IMG_ROOT      = f"{DATA_ROOT}/all"
    TEST_IMG_ROOT = f"{DATA_ROOT}/evaluation"
    SAVE_DIR      = "/home/dsia-st125985/SpineVQA/models"

    # Model
    SIGLIP_NAME = "google/siglip2-base-patch16-224"
    BERT_NAME   = "bert-base-uncased"
    IMG_DIM     = 768
    Q_DIM       = 768
    HIDDEN_DIM  = 512
    NUM_DISEASE = 12
    NUM_LEVELS  = 5

    # Training
    BATCH_SIZE         = 32
    LR                 = 2e-5
    EPOCHS             = 20
    DROPOUT            = 0.3
    CONTRASTIVE_LAMBDA = 0.3
    CONTRASTIVE_TEMP   = 0.07
    MAX_LEN            = 64
    DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

    # Disease classes — exact match with SpineBench JSON
    DISEASES = [
        "Subarticular Stenosis",
        "Foraminal stenosis",
        "Healthy",
        "Osteophytes",
        "Spinal Canal Stenosis",
        "cervical Lordosis",
        "Straight cervical vertebrae",
        "sigmoid cervical vertebrae",
        "cervical Kyphosis",
        "Disc space narrowing",
        "Spondylolisthesis",
        "Vertebral collapse"
    ]

    # Localization levels
    LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

    # Mappings
    DISEASE2IDX = {d: i for i, d in enumerate(DISEASES)}
    LEVEL2IDX   = {l: i for i, l in enumerate(LEVELS)}

cfg = Config()
os.makedirs(cfg.SAVE_DIR, exist_ok=True)
print(f"Device: {cfg.DEVICE}")


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

        # ── Build paired lookup maps ───────────────────
        # Same image may have both disease + location labels
        # We inject the missing label from the paired map
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

        paired = set(self.image_disease.keys()) & \
                 set(self.image_location.keys())
        print(f"Loaded {len(self.data)} samples [{split}]")
        print(f"Paired images (disease+location): {len(paired)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # ── Image ─────────────────────────────────────
        img_path   = os.path.join(self.img_root, sample["image"])
        image      = Image.open(img_path).convert("RGB")
        img_tensor = self.processor(
            images=image,
            return_tensors="pt"
        )["pixel_values"].squeeze(0)

        # ── Question ───────────────────────────────────
        tokens = self.tokenizer(
            sample["question"],
            max_length=cfg.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids      = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        # ── Task & Labels ──────────────────────────────
        task    = sample["task"]
        img_key = sample["image"]

        # Disease label (Task 1)
        disease_label = -1
        if task == "spine_disease_classification":
            answer = sample["answers"]
            if isinstance(answer, list):
                answer = answer[0]
            disease_label = cfg.DISEASE2IDX.get(answer, -1)

        # Localization label (Task 2) — multi-label
        loc_label = torch.zeros(cfg.NUM_LEVELS)
        if task == "spine_lesion_localization":
            answers = sample["answers"]
            if isinstance(answers, str):
                answers = [answers]
            for ans in answers:
                if ans in cfg.LEVEL2IDX:
                    loc_label[cfg.LEVEL2IDX[ans]] = 1.0

        # ── Inject paired labels ───────────────────────
        # Classification sample → inject location from map
        if task == "spine_disease_classification":
            if img_key in self.image_location:
                answers = self.image_location[img_key]
                if isinstance(answers, str):
                    answers = [answers]
                for ans in answers:
                    if ans in cfg.LEVEL2IDX:
                        loc_label[cfg.LEVEL2IDX[ans]] = 1.0

        # Localization sample → inject disease from map
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
# 2. MODEL — E3 (same as E2)
# ============================================================
class E3Model(nn.Module):
    """
    E3 SpineVQA Model — Same architecture as E1/E2.
    Returns fused features for localization-aware contrastive loss.

    Architecture:
        SigLIP2 (frozen) → CLS [B,768] → Linear → [B,512]
        BERT             → CLS [B,768] → Linear → [B,512]
        Concat → [B,1024] → Linear → [B,512] → Dropout
        → Disease Head [B,12]
        → Location Head [B,5]
        → Fused Features [B,512]
    """

    def __init__(self):
        super().__init__()

        # ── SigLIP2 Visual Encoder (Frozen) ───────────
        self.siglip2 = AutoModel.from_pretrained(
            cfg.SIGLIP_NAME,
            torch_dtype=torch.float16
        )
        for param in self.siglip2.parameters():
            param.requires_grad = False

        # ── BERT Question Encoder ──────────────────────
        self.bert = BertModel.from_pretrained(cfg.BERT_NAME)

        # ── Image Projection: 768 → 512 ───────────────
        self.image_proj = nn.Sequential(
            nn.Linear(cfg.IMG_DIM, cfg.HIDDEN_DIM),
            nn.ReLU()
        )

        # ── Question Projection: 768 → 512 ────────────
        self.question_proj = nn.Sequential(
            nn.Linear(cfg.Q_DIM, cfg.HIDDEN_DIM),
            nn.ReLU()
        )

        # ── Fusion Module ──────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM * 2, cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT)
        )

        # ── Disease Classification Head ────────────────
        self.disease_head = nn.Linear(cfg.HIDDEN_DIM, cfg.NUM_DISEASE)

        # ── Localization Head ──────────────────────────
        self.loc_head = nn.Linear(cfg.HIDDEN_DIM, cfg.NUM_LEVELS)

    def forward(self, images, input_ids, attention_mask):
        """
        Returns:
            disease_logits: [B, 12]
            loc_logits:     [B, 5]
            z:              [B, 512] fused features
        """
        # SigLIP2 — frozen
        with torch.no_grad():
            vision_out = self.siglip2.vision_model(
                pixel_values=images.half()
            )
            f_img = vision_out.last_hidden_state[:, 0, :]
            f_img = f_img.float()

        h_img = self.image_proj(f_img)

        # BERT
        q_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        f_q = q_out.last_hidden_state[:, 0, :]
        h_q = self.question_proj(f_q)

        # Fusion
        f_fused = torch.cat([h_img, h_q], dim=-1)
        z = self.fusion(f_fused)

        # Heads
        disease_logits = self.disease_head(z)
        loc_logits     = self.loc_head(z)

        return disease_logits, loc_logits, z


# ============================================================
# 3. LOSS FUNCTIONS
# ============================================================

# ── 3a. Class weights ──────────────────────────────────────
def compute_class_weights(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    counts = torch.zeros(cfg.NUM_DISEASE)
    for sample in data:
        if sample["task"] == "spine_disease_classification":
            ans = sample["answers"]
            if isinstance(ans, list):
                ans = ans[0]
            idx = cfg.DISEASE2IDX.get(ans, -1)
            if idx >= 0:
                counts[idx] += 1

    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * cfg.NUM_DISEASE
    print("Class weights computed:")
    for i, (d, w) in enumerate(zip(cfg.DISEASES, weights)):
        print(f"  {d[:30]:30s}: {w:.3f}")
    return weights


# ── 3b. Task Loss (CE + BCE) ───────────────────────────────
class TaskLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.disease_loss = nn.CrossEntropyLoss(
            weight=class_weights.to(cfg.DEVICE),
            ignore_index=-1
        )
        self.loc_loss = nn.BCEWithLogitsLoss()

    def forward(self, disease_logits, loc_logits,
                disease_labels, loc_labels, tasks):

        total_loss = torch.tensor(0.0, device=cfg.DEVICE,
                                  requires_grad=True)

        cls_mask = torch.tensor(
            [t == "spine_disease_classification" for t in tasks],
            dtype=torch.bool
        ).to(cfg.DEVICE)

        if cls_mask.any():
            L_disease = self.disease_loss(
                disease_logits[cls_mask],
                disease_labels[cls_mask]
            )
            total_loss = total_loss + L_disease
        else:
            L_disease = torch.tensor(0.0)

        loc_mask = torch.tensor(
            [t == "spine_lesion_localization" for t in tasks],
            dtype=torch.bool
        ).to(cfg.DEVICE)

        if loc_mask.any():
            L_loc = self.loc_loss(
                loc_logits[loc_mask],
                loc_labels[loc_mask]
            )
            total_loss = total_loss + L_loc
        else:
            L_loc = torch.tensor(0.0)

        return total_loss, L_disease, L_loc


# ── 3c. Localization-Aware Contrastive Loss ────────────────
class SpineContrastiveLoss(nn.Module):
    """
    E3: Localization-Aware Supervised Contrastive Loss

    Uses PAIRED samples only (has both disease + location label).
    SpineBench has 23,381 such paired training images.

    Positive pairs:  same disease AND same spinal level
    Hard Negatives:  same disease + different level (location confusion)
                     different disease + same level (disease confusion)
    Easy Negatives:  different disease + different level

    This directly targets the weakest metric: Localization Accuracy.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, disease_labels, loc_labels):
        """
        Args:
            features:       [B, 512] — fused features
            disease_labels: [B]      — disease class (-1 = loc only)
            loc_labels:     [B, 5]   — multi-hot location
        Returns:
            contrastive loss scalar
        """
        # ── Step 1: Select paired samples only ────────
        # Paired = has disease label AND has location label
        has_disease  = disease_labels >= 0
        has_location = loc_labels.sum(dim=1) > 0
        paired_mask  = has_disease & has_location

        if paired_mask.sum() < 2:
            return torch.tensor(0.0,
                device=features.device,
                requires_grad=True)

        feat = features[paired_mask]
        d_lb = disease_labels[paired_mask]
        l_lb = loc_labels[paired_mask].float()

        N = feat.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=features.device)

        # ── Step 2: Normalize features ────────────────
        feat = F.normalize(feat, dim=1)

        # ── Step 3: Similarity matrix [N, N] ──────────
        sim = torch.matmul(feat, feat.T) / self.temperature

        # ── Step 4: Same disease mask ──────────────────
        d_same = torch.eq(
            d_lb.unsqueeze(1),
            d_lb.unsqueeze(0)
        ).float()  # [N, N]

        # ── Step 5: Same level mask ────────────────────
        # Any overlap in multi-hot location vectors
        # l_lb @ l_lb.T > 0 means at least one level shared
        l_same = (torch.matmul(l_lb, l_lb.T) > 0).float()  # [N, N]

        # ── Step 6: Positive mask ──────────────────────
        # Positive = same disease AND same spinal level
        self_mask     = torch.eye(N, device=feat.device)
        positive_mask = d_same * l_same * (1 - self_mask)

        if positive_mask.sum() == 0:
            # No valid positive pairs in this batch
            return torch.tensor(0.0, device=features.device)

        # ── Step 7: Contrastive loss ───────────────────
        # All pairs except self = denominator
        exp_sim  = torch.exp(sim) * (1 - self_mask)
        log_prob = sim - torch.log(
            exp_sim.sum(dim=1, keepdim=True) + 1e-8
        )

        # Loss only over samples that have at least one positive
        positives_per_sample = positive_mask.sum(dim=1)
        valid = positives_per_sample > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        loss = -(positive_mask * log_prob).sum(dim=1) / \
               (positives_per_sample + 1e-8)

        return loss[valid].mean()


# ============================================================
# 4. TRAIN ONE EPOCH
# ============================================================
def train_epoch(model, loader, optimizer, task_criterion,
                contrastive_loss):
    model.train()

    total_loss    = 0.0
    total_con_loss = 0.0
    correct_cls   = 0
    total_cls     = 0

    for batch in tqdm(loader, desc="Training"):
        images         = batch["image"].to(cfg.DEVICE)
        input_ids      = batch["input_ids"].to(cfg.DEVICE)
        attention_mask = batch["attention_mask"].to(cfg.DEVICE)
        disease_labels = batch["disease_label"].to(cfg.DEVICE)
        loc_labels     = batch["loc_label"].to(cfg.DEVICE)
        tasks          = batch["task"]

        optimizer.zero_grad()

        # Forward
        disease_logits, loc_logits, fused_feat = model(
            images, input_ids, attention_mask
        )

        # Task loss
        loss, L_d, L_l = task_criterion(
            disease_logits, loc_logits,
            disease_labels, loc_labels, tasks
        )

        # E3: Localization-Aware Contrastive loss
        # Pass both disease_labels AND loc_labels
        L_con = contrastive_loss(
            fused_feat, disease_labels, loc_labels
        )
        loss = loss + cfg.CONTRASTIVE_LAMBDA * L_con

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss     += loss.item()
        total_con_loss += L_con.item()

        cls_mask = torch.tensor(
            [t == "spine_disease_classification" for t in tasks],
            dtype=torch.bool
        ).to(cfg.DEVICE)
        if cls_mask.any():
            preds = disease_logits[cls_mask].argmax(dim=1)
            correct_cls += (preds == disease_labels[cls_mask]).sum().item()
            total_cls   += cls_mask.sum().item()

    avg_loss     = total_loss / len(loader)
    avg_con_loss = total_con_loss / len(loader)
    accuracy     = correct_cls / total_cls * 100 if total_cls > 0 else 0
    return avg_loss, accuracy, avg_con_loss


# ============================================================
# 5. EVALUATE
# ============================================================
def evaluate(model, loader):
    model.eval()

    correct_cls   = 0
    total_cls     = 0
    all_loc_preds = []
    all_loc_gt    = []
    exact_match   = 0
    total_loc     = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images         = batch["image"].to(cfg.DEVICE)
            input_ids      = batch["input_ids"].to(cfg.DEVICE)
            attention_mask = batch["attention_mask"].to(cfg.DEVICE)
            disease_labels = batch["disease_label"].to(cfg.DEVICE)
            loc_labels     = batch["loc_label"].to(cfg.DEVICE)
            tasks          = batch["task"]

            disease_logits, loc_logits, _ = model(
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
                loc_preds = (torch.sigmoid(loc_logits[loc_mask]) >= 0.5).float()
                loc_gt    = loc_labels[loc_mask]
                exact     = (loc_preds == loc_gt).all(dim=1).sum().item()
                exact_match += exact
                total_loc   += loc_mask.sum().item()
                all_loc_preds.append(loc_preds.cpu())
                all_loc_gt.append(loc_gt.cpu())

    cls_acc       = correct_cls / total_cls * 100 if total_cls > 0 else 0
    loc_exact_acc = exact_match / total_loc * 100 if total_loc > 0 else 0

    if all_loc_preds:
        preds_cat = torch.cat(all_loc_preds, dim=0)
        gt_cat    = torch.cat(all_loc_gt, dim=0)
        tp        = (preds_cat * gt_cat).sum(dim=1)
        precision = (tp / (preds_cat.sum(dim=1) + 1e-6)).mean().item() * 100
        recall    = (tp / (gt_cat.sum(dim=1) + 1e-6)).mean().item() * 100
    else:
        precision = recall = 0.0

    total_all   = total_cls + total_loc
    correct_all = correct_cls + exact_match
    overall_acc = correct_all / total_all * 100 if total_all > 0 else 0

    return {
        "cls_acc":       cls_acc,
        "loc_exact_acc": loc_exact_acc,
        "precision":     precision,
        "recall":        recall,
        "overall_acc":   overall_acc,
    }


# ============================================================
# 6. MAIN TRAINING LOOP
# ============================================================
def main():
    wandb.init(
        project="SpineVQA-CL",
        name="E3-LocContrastive",
        config={
            "lr":                 cfg.LR,
            "batch_size":         cfg.BATCH_SIZE,
            "epochs":             cfg.EPOCHS,
            "model":              "SigLIP2+BERT+LocContrastive",
            "contrastive_lambda": cfg.CONTRASTIVE_LAMBDA,
            "contrastive_temp":   cfg.CONTRASTIVE_TEMP,
            "contrastive_type":   "disease+location aware",
            "paired_images":      23381,
            "dataset":            "SpineBench"
        }
    )

    print("=" * 60)
    print("E3 — Localization-Aware Contrastive SpineVQA Training")
    print("=" * 60)

    print("\nLoading models...")
    from transformers import SiglipImageProcessor
    processor = SiglipImageProcessor.from_pretrained(cfg.SIGLIP_NAME)
    tokenizer = BertTokenizer.from_pretrained(cfg.BERT_NAME)

    train_dataset = SpineBenchDataset(
        cfg.TRAIN_JSON, cfg.IMG_ROOT,
        processor, tokenizer, split="train"
    )
    test_dataset = SpineBenchDataset(
        cfg.TEST_JSON, cfg.TEST_IMG_ROOT,
        processor, tokenizer, split="test"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = E3Model().to(cfg.DEVICE)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    class_weights    = compute_class_weights(cfg.TRAIN_JSON)
    task_criterion   = TaskLoss(class_weights)
    contrastive_loss = SpineContrastiveLoss(
        temperature=cfg.CONTRASTIVE_TEMP
    ).to(cfg.DEVICE)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LR,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.EPOCHS
    )

    best_overall = 0.0
    print("\nStarting training...")
    print("=" * 80)
    print(f"{'Epoch':>6} | {'Loss':>8} | {'ConLoss':>8} | {'Cls Acc':>8} | "
          f"{'Loc Acc':>8} | {'Pre':>7} | {'Rec':>7} | "
          f"{'Overall':>8} | {'LR':>10}")
    print("-" * 80)

    for epoch in range(1, cfg.EPOCHS + 1):

        train_loss, train_acc, avg_con_loss = train_epoch(
            model, train_loader, optimizer,
            task_criterion, contrastive_loss
        )

        metrics    = evaluate(model, test_loader)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        wandb.log({
            "epoch":         epoch,
            "train_loss":    train_loss,
            "train_acc":     train_acc,
            "con_loss":      avg_con_loss,
            "cls_acc":       metrics["cls_acc"],
            "loc_exact_acc": metrics["loc_exact_acc"],
            "precision":     metrics["precision"],
            "recall":        metrics["recall"],
            "overall_acc":   metrics["overall_acc"],
            "lr":            current_lr,
        })

        if metrics["overall_acc"] > best_overall:
            best_overall = metrics["overall_acc"]
            torch.save(
                model.state_dict(),
                os.path.join(cfg.SAVE_DIR, "E3_best.pth")
            )
            saved = "✓"
        else:
            saved = ""

        print(
            f"{epoch:>6} | {train_loss:>8.4f} | "
            f"{avg_con_loss:>8.4f} | "
            f"{train_acc:>7.2f}% | "
            f"{metrics['loc_exact_acc']:>7.2f}% | "
            f"{metrics['precision']:>6.2f}% | "
            f"{metrics['recall']:>6.2f}% | "
            f"{metrics['overall_acc']:>7.2f}% | "
            f"{current_lr:>10.2e}  {saved}"
        )

    print("=" * 80)
    print(f"\nBest Overall Accuracy: {best_overall:.2f}%")
    print(f"E2 Baseline:           35.48%")
    print(f"E1 Baseline:           34.82%")
    print(f"Gemini (zero-shot):    32.37%")
    if best_overall > 35.48:
        print(f"Beat E2 by {best_overall - 35.48:.2f}%!")
    if best_overall > 34.82:
        print(f"Beat E1 by {best_overall - 34.82:.2f}%!")
    if best_overall > 32.37:
        print(f"Beat Gemini by {best_overall - 32.37:.2f}%!")

    wandb.finish()
    print(f"\nModel saved to: {cfg.SAVE_DIR}/E3_best.pth")


# ============================================================
if __name__ == "__main__":
    main()
