# ============================================================
# E2 — Contrastive SpineVQA Model
# Thesis: Localization-Aware Fine-Grained Contrastive Learning
#         for Spinal Pathology Representation in Medical VQA
# Dataset: SpineBench (ACM MM 2025)
# Change from E1: Added SpineContrastiveLoss
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
        print(f"Loaded {len(self.data)} samples [{split}]")

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
        task = sample["task"]

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

        return {
            "image":          img_tensor,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "task":           task,
            "disease_label":  torch.tensor(disease_label, dtype=torch.long),
            "loc_label":      loc_label,
        }


# ============================================================
# 2. MODEL — E2 (same as E1, returns fused features)
# ============================================================
class E2Model(nn.Module):
    """
    E2 SpineVQA Model — Same as E1 but returns fused features
    for contrastive learning.

    Architecture:
        SigLIP2 (Image) → CLS [B,768] → Linear → [B,512]
        BERT   (Question)→ CLS [B,768] → Linear → [B,512]
        Concat → [B,1024] → Linear → [B,512] → Dropout
        → Disease Head [B,12]
        → Location Head [B,5]
        → Fused Features [B,512]  ← NEW: for contrastive loss
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
            z:              [B, 512]  ← fused features for contrastive
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
    """Compute inverse frequency weights for disease classes."""
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

        # Task 1 — Disease Classification
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

        # Task 2 — Localization
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


# ── 3c. Spine Contrastive Loss ─────────────────────────────
class SpineContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss for Spine Disease Classification.

    Positive pairs:  same disease class
    Negatives:       different disease class

    Only uses classification samples (disease_label >= 0).
    Simple and clean — proves contrastive learning benefit in E2.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: [B, 512] — fused features
            labels:   [B]      — disease labels (-1 = localization)
        Returns:
            contrastive loss scalar
        """
        # Use only classification samples
        mask     = labels >= 0
        features = features[mask]
        labels   = labels[mask]

        if features.size(0) < 2:
            return torch.tensor(0.0, device=features.device)

        # Normalize
        features = F.normalize(features, dim=1)

        # Similarity matrix [N, N]
        sim = torch.matmul(features, features.T) / self.temperature

        # Positive mask: same disease label
        labels        = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float().to(features.device)

        # Remove diagonal (self-similarity)
        self_mask     = torch.eye(features.size(0), device=features.device)
        positive_mask = positive_mask - self_mask

        # Log probability
        exp_sim  = torch.exp(sim) * (1 - self_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Loss over positive pairs
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

        # Forward — model returns 3 values
        disease_logits, loc_logits, fused_feat = model(
            images, input_ids, attention_mask
        )

        # Task loss (CE + BCE)
        loss, L_d, L_l = task_criterion(
            disease_logits, loc_logits,
            disease_labels, loc_labels, tasks
        )

        # Contrastive loss (disease-level supervised)
        L_con = contrastive_loss(fused_feat, disease_labels)
        loss  = loss + cfg.CONTRASTIVE_LAMBDA * L_con

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        # Accuracy (classification only)
        cls_mask = torch.tensor(
            [t == "spine_disease_classification" for t in tasks],
            dtype=torch.bool
        ).to(cfg.DEVICE)
        if cls_mask.any():
            preds = disease_logits[cls_mask].argmax(dim=1)
            correct_cls += (preds == disease_labels[cls_mask]).sum().item()
            total_cls   += cls_mask.sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct_cls / total_cls * 100 if total_cls > 0 else 0
    return avg_loss, accuracy


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

            # Model returns 3 values — use _ for fused features
            disease_logits, loc_logits, _ = model(
                images, input_ids, attention_mask
            )

            # Task 1 accuracy
            cls_mask = torch.tensor(
                [t == "spine_disease_classification" for t in tasks],
                dtype=torch.bool
            ).to(cfg.DEVICE)
            if cls_mask.any():
                preds = disease_logits[cls_mask].argmax(dim=1)
                correct_cls += (preds == disease_labels[cls_mask]).sum().item()
                total_cls   += cls_mask.sum().item()

            # Task 2 — exact match
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
        name="E2-Contrastive",
        config={
            "lr":                 cfg.LR,
            "batch_size":         cfg.BATCH_SIZE,
            "epochs":             cfg.EPOCHS,
            "model":              "SigLIP2+BERT+Contrastive",
            "contrastive_lambda": cfg.CONTRASTIVE_LAMBDA,
            "contrastive_temp":   cfg.CONTRASTIVE_TEMP,
            "dataset":            "SpineBench"
        }
    )

    print("=" * 60)
    print("E2 — Contrastive SpineVQA Training")
    print("=" * 60)

    # ── Load Processors ───────────────────────────────
    print("\nLoading models...")
    from transformers import SiglipImageProcessor
    processor = SiglipImageProcessor.from_pretrained(cfg.SIGLIP_NAME)
    tokenizer = BertTokenizer.from_pretrained(cfg.BERT_NAME)

    # ── Datasets ──────────────────────────────────────
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

    # ── Model ─────────────────────────────────────────
    model = E2Model().to(cfg.DEVICE)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ── Loss Functions ────────────────────────────────
    class_weights    = compute_class_weights(cfg.TRAIN_JSON)
    task_criterion   = TaskLoss(class_weights)
    contrastive_loss = SpineContrastiveLoss(
        temperature=cfg.CONTRASTIVE_TEMP
    ).to(cfg.DEVICE)

    # ── Optimizer ─────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LR,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.EPOCHS
    )

    # ── Training Loop ─────────────────────────────────
    best_overall = 0.0
    print("\nStarting training...")
    print("=" * 80)
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Cls Acc':>8} | "
          f"{'Loc Acc':>8} | {'Pre':>7} | {'Rec':>7} | "
          f"{'Overall':>8} | {'LR':>10}")
    print("-" * 80)

    for epoch in range(1, cfg.EPOCHS + 1):

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer,
            task_criterion, contrastive_loss
        )

        metrics    = evaluate(model, test_loader)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # WandB log
        wandb.log({
            "epoch":         epoch,
            "train_loss":    train_loss,
            "train_acc":     train_acc,
            "cls_acc":       metrics["cls_acc"],
            "loc_exact_acc": metrics["loc_exact_acc"],
            "precision":     metrics["precision"],
            "recall":        metrics["recall"],
            "overall_acc":   metrics["overall_acc"],
            "lr":            current_lr,
        })

        # Save best
        if metrics["overall_acc"] > best_overall:
            best_overall = metrics["overall_acc"]
            torch.save(
                model.state_dict(),
                os.path.join(cfg.SAVE_DIR, "E2_best.pth")
            )
            saved = "✓"
        else:
            saved = ""

        print(
            f"{epoch:>6} | {train_loss:>8.4f} | "
            f"{train_acc:>7.2f}% | "
            f"{metrics['loc_exact_acc']:>7.2f}% | "
            f"{metrics['precision']:>6.2f}% | "
            f"{metrics['recall']:>6.2f}% | "
            f"{metrics['overall_acc']:>7.2f}% | "
            f"{current_lr:>10.2e}  {saved}"
        )

    print("=" * 80)
    print(f"\nBest Overall Accuracy: {best_overall:.2f}%")
    print(f"E1 Baseline:           34.82%")
    print(f"SpineBench Best (Gemini zero-shot): 32.37%")
    if best_overall > 34.82:
        print(f"✅ Beat E1 by {best_overall - 34.82:.2f}%!")
    if best_overall > 32.37:
        print(f"✅ Beat Gemini by {best_overall - 32.37:.2f}%!")

    wandb.finish()
    print(f"\nModel saved to: {cfg.SAVE_DIR}/E2_best.pth")


# ============================================================
if __name__ == "__main__":
    main()
