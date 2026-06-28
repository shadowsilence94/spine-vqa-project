# ============================================================
# E1a — Baseline SpineVQA Model with Proper Train/Val/Test Split
# Thesis: Localization-Aware Fine-Grained Contrastive Learning
#         for Spinal Pathology Representation in Medical VQA
#
# E1a meaning:
#   Same E1 baseline architecture:
#       SigLIP2 CLS token + BERT CLS token + simple fusion
#
#   Updated protocol:
#       train_split.json for training
#       val_split.json for best epoch selection
#       test.json evaluated only once at the end
# ============================================================

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from transformers import BertTokenizer, BertModel
from transformers import SiglipImageProcessor
from PIL import Image
from tqdm import tqdm
import wandb


# ============================================================
# 0. CONFIG
# ============================================================

class Config:
    # Paths
    DATA_ROOT = "/home/dsia-st125985/SpineVQA/data/SpineBench"

    TRAIN_JSON = f"{DATA_ROOT}/all/train_split.json"
    VAL_JSON = f"{DATA_ROOT}/all/val_split.json"
    TEST_JSON = f"{DATA_ROOT}/evaluation/test.json"

    TRAIN_IMG_ROOT = f"{DATA_ROOT}/all"
    VAL_IMG_ROOT = f"{DATA_ROOT}/all"
    TEST_IMG_ROOT = f"{DATA_ROOT}/evaluation"

    SAVE_DIR = "/home/dsia-st125985/SpineVQA/models"

    # Model
    SIGLIP_NAME = "google/siglip2-base-patch16-224"
    BERT_NAME = "bert-base-uncased"

    IMG_DIM = 768
    Q_DIM = 768
    HIDDEN_DIM = 512

    NUM_DISEASE = 12
    NUM_LEVELS = 5

    # Training
    BATCH_SIZE = 32
    LR = 2e-5
    EPOCHS = 20
    DROPOUT = 0.3
    MAX_LEN = 64
    NUM_WORKERS = 4

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Disease classes: exact names from SpineBench JSON
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
        "Vertebral collapse",
    ]

    # Localization levels
    LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

    DISEASE2IDX = {d: i for i, d in enumerate(DISEASES)}
    LEVEL2IDX = {l: i for i, l in enumerate(LEVELS)}


cfg = Config()
os.makedirs(cfg.SAVE_DIR, exist_ok=True)
print(f"Device: {cfg.DEVICE}")


# ============================================================
# 1. DATASET
# ============================================================

class SpineBenchDataset(Dataset):
    def __init__(self, json_path, img_root, processor, tokenizer, split="train"):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.img_root = img_root
        self.processor = processor
        self.tokenizer = tokenizer
        self.split = split

        print(f"Loaded {len(self.data):,} samples [{split}]")

    def __len__(self):
        return len(self.data)

    def resolve_image_path(self, sample_image):
        """
        Robust image path resolver.

        Why needed:
        - Some JSON entries may store image as "11.jpg"
        - Some may store image as "folder/11.jpg"
        - Train/val images are under SpineBench/all
        - Test images are under SpineBench/evaluation
        """

        candidate_paths = [
            os.path.join(self.img_root, sample_image),
            os.path.join(self.img_root, os.path.basename(sample_image)),
            os.path.join(cfg.DATA_ROOT, sample_image),
            os.path.join(cfg.DATA_ROOT, "all", sample_image),
            os.path.join(cfg.DATA_ROOT, "all", os.path.basename(sample_image)),
            os.path.join(cfg.DATA_ROOT, "evaluation", sample_image),
            os.path.join(cfg.DATA_ROOT, "evaluation", os.path.basename(sample_image)),
        ]

        for path in candidate_paths:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            f"Image not found for sample image: {sample_image}\n"
            f"Tried paths:\n" + "\n".join(candidate_paths)
        )

    def __getitem__(self, idx):
        sample = self.data[idx]

        # ---------------- Image ----------------
        img_path = self.resolve_image_path(sample["image"])

        image = Image.open(img_path).convert("RGB")

        img_tensor = self.processor(
            images=image,
            return_tensors="pt"
        )["pixel_values"].squeeze(0)

        # ---------------- Question ----------------
        tokens = self.tokenizer(
            sample["question"],
            max_length=cfg.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        # ---------------- Labels ----------------
        task = sample["task"]

        # SpineBench sometimes uses "answer", older code used "answers"
        answer = sample.get("answer", sample.get("answers", ""))

        disease_label = -1
        loc_label = torch.zeros(cfg.NUM_LEVELS, dtype=torch.float32)

        # Disease classification
        if task == "spine_disease_classification":
            if isinstance(answer, list):
                answer = answer[0]

            disease_label = cfg.DISEASE2IDX.get(answer, -1)

        # Lesion localization
        elif task == "spine_lesion_localization":
            if isinstance(answer, str):
                answer = [answer]

            for ans in answer:
                if ans in cfg.LEVEL2IDX:
                    loc_label[cfg.LEVEL2IDX[ans]] = 1.0

        return {
            "image": img_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "task": task,
            "disease_label": torch.tensor(disease_label, dtype=torch.long),
            "loc_label": loc_label,
        }


# ============================================================
# 2. MODEL — E1a BASELINE
# ============================================================

class E1aBaselineModel(nn.Module):
    """
    E1a Baseline SpineVQA Model

    Architecture:
        SigLIP2 image encoder → CLS token [B,768]
        BERT question encoder → CLS token [B,768]
        Image projection      → [B,512]
        Question projection   → [B,512]
        Concat                → [B,1024]
        Fusion                → [B,512]
        Disease head          → [B,12]
        Location head         → [B,5]

    Difference from E1:
        Architecture is same.
        Training protocol is corrected using train/val/test.
    """

    def __init__(self):
        super().__init__()

        # SigLIP2 visual encoder
        self.siglip2 = AutoModel.from_pretrained(
            cfg.SIGLIP_NAME,
            torch_dtype=torch.float16
        )

        # Freeze SigLIP2
        for param in self.siglip2.parameters():
            param.requires_grad = False

        # BERT question encoder
        self.bert = BertModel.from_pretrained(cfg.BERT_NAME)

        self.image_proj = nn.Sequential(
            nn.Linear(cfg.IMG_DIM, cfg.HIDDEN_DIM),
            nn.ReLU()
        )

        self.question_proj = nn.Sequential(
            nn.Linear(cfg.Q_DIM, cfg.HIDDEN_DIM),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM * 2, cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT)
        )

        self.disease_head = nn.Linear(cfg.HIDDEN_DIM, cfg.NUM_DISEASE)
        self.loc_head = nn.Linear(cfg.HIDDEN_DIM, cfg.NUM_LEVELS)

    def forward(self, images, input_ids, attention_mask):
        # ---------------- Image encoder ----------------
        with torch.no_grad():
            vision_out = self.siglip2.vision_model(
                pixel_values=images.half()
            )

            # E1 baseline uses only CLS token
            f_img = vision_out.last_hidden_state[:, 0, :]
            f_img = f_img.float()

        h_img = self.image_proj(f_img)

        # ---------------- Question encoder ----------------
        q_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        f_q = q_out.last_hidden_state[:, 0, :]
        h_q = self.question_proj(f_q)

        # ---------------- Fusion ----------------
        fused = torch.cat([h_img, h_q], dim=-1)
        z = self.fusion(fused)

        disease_logits = self.disease_head(z)
        loc_logits = self.loc_head(z)

        return disease_logits, loc_logits


# ============================================================
# 3. CLASS WEIGHTS
# ============================================================

def compute_class_weights(json_path):
    """
    Compute inverse-frequency weights for disease classes
    using only train_split.json.
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    counts = torch.zeros(cfg.NUM_DISEASE)

    for sample in data:
        if sample["task"] == "spine_disease_classification":
            answer = sample.get("answer", sample.get("answers", ""))

            if isinstance(answer, list):
                answer = answer[0]

            idx = cfg.DISEASE2IDX.get(answer, -1)

            if idx >= 0:
                counts[idx] += 1

    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * cfg.NUM_DISEASE

    print("Class counts:")
    for disease_name, count in zip(cfg.DISEASES, counts):
        print(f"  {disease_name:30s}: {int(count.item())}")

    print("\nClass weights computed:")
    for disease_name, weight in zip(cfg.DISEASES, weights):
        print(f"  {disease_name:30s}: {weight:.3f}")

    return weights


# ============================================================
# 4. LOSS
# ============================================================

class E1aLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()

        self.disease_loss = nn.CrossEntropyLoss(
            weight=class_weights.to(cfg.DEVICE),
            ignore_index=-1
        )

        self.loc_loss = nn.BCEWithLogitsLoss()

    def forward(self, disease_logits, loc_logits, disease_labels, loc_labels, tasks):
        total_loss = torch.tensor(
            0.0,
            device=cfg.DEVICE,
            requires_grad=True
        )

        # Disease classification samples
        cls_mask = torch.tensor(
            [t == "spine_disease_classification" for t in tasks],
            dtype=torch.bool,
            device=cfg.DEVICE
        )

        if cls_mask.any():
            loss_disease = self.disease_loss(
                disease_logits[cls_mask],
                disease_labels[cls_mask]
            )
            total_loss = total_loss + loss_disease
        else:
            loss_disease = torch.tensor(0.0, device=cfg.DEVICE)

        # Localization samples
        loc_mask = torch.tensor(
            [t == "spine_lesion_localization" for t in tasks],
            dtype=torch.bool,
            device=cfg.DEVICE
        )

        if loc_mask.any():
            loss_loc = self.loc_loss(
                loc_logits[loc_mask],
                loc_labels[loc_mask]
            )
            total_loss = total_loss + loss_loc
        else:
            loss_loc = torch.tensor(0.0, device=cfg.DEVICE)

        return total_loss, loss_disease, loss_loc


# ============================================================
# 5. TRAIN ONE EPOCH
# ============================================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()

    total_loss = 0.0

    correct_cls = 0
    total_cls = 0

    for batch in tqdm(loader, desc="Training"):
        images = batch["image"].to(cfg.DEVICE)
        input_ids = batch["input_ids"].to(cfg.DEVICE)
        attention_mask = batch["attention_mask"].to(cfg.DEVICE)
        disease_labels = batch["disease_label"].to(cfg.DEVICE)
        loc_labels = batch["loc_label"].to(cfg.DEVICE)
        tasks = batch["task"]

        optimizer.zero_grad()

        disease_logits, loc_logits = model(
            images,
            input_ids,
            attention_mask
        )

        loss, loss_disease, loss_loc = criterion(
            disease_logits,
            loc_logits,
            disease_labels,
            loc_labels,
            tasks
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        cls_mask = torch.tensor(
            [t == "spine_disease_classification" for t in tasks],
            dtype=torch.bool,
            device=cfg.DEVICE
        )

        if cls_mask.any():
            preds = disease_logits[cls_mask].argmax(dim=1)
            correct_cls += (preds == disease_labels[cls_mask]).sum().item()
            total_cls += cls_mask.sum().item()

    avg_loss = total_loss / len(loader)
    train_cls_acc = correct_cls / total_cls * 100 if total_cls > 0 else 0.0

    return avg_loss, train_cls_acc


# ============================================================
# 6. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    correct_cls = 0
    total_cls = 0

    exact_match = 0
    total_loc = 0

    all_loc_preds = []
    all_loc_gt = []

    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].to(cfg.DEVICE)
        input_ids = batch["input_ids"].to(cfg.DEVICE)
        attention_mask = batch["attention_mask"].to(cfg.DEVICE)
        disease_labels = batch["disease_label"].to(cfg.DEVICE)
        loc_labels = batch["loc_label"].to(cfg.DEVICE)
        tasks = batch["task"]

        disease_logits, loc_logits = model(
            images,
            input_ids,
            attention_mask
        )

        # Disease classification
        cls_mask = torch.tensor(
            [t == "spine_disease_classification" for t in tasks],
            dtype=torch.bool,
            device=cfg.DEVICE
        )

        if cls_mask.any():
            preds = disease_logits[cls_mask].argmax(dim=1)
            correct_cls += (preds == disease_labels[cls_mask]).sum().item()
            total_cls += cls_mask.sum().item()

        # Localization
        loc_mask = torch.tensor(
            [t == "spine_lesion_localization" for t in tasks],
            dtype=torch.bool,
            device=cfg.DEVICE
        )

        if loc_mask.any():
            loc_preds = (torch.sigmoid(loc_logits[loc_mask]) >= 0.5).float()
            loc_gt = loc_labels[loc_mask]

            exact = (loc_preds == loc_gt).all(dim=1).sum().item()
            exact_match += exact
            total_loc += loc_mask.sum().item()

            all_loc_preds.append(loc_preds.cpu())
            all_loc_gt.append(loc_gt.cpu())

    cls_acc = correct_cls / total_cls * 100 if total_cls > 0 else 0.0
    loc_exact_acc = exact_match / total_loc * 100 if total_loc > 0 else 0.0

    if all_loc_preds:
        preds_cat = torch.cat(all_loc_preds, dim=0)
        gt_cat = torch.cat(all_loc_gt, dim=0)

        tp = (preds_cat * gt_cat).sum(dim=1)

        precision = (
            tp / (preds_cat.sum(dim=1) + 1e-6)
        ).mean().item() * 100

        recall = (
            tp / (gt_cat.sum(dim=1) + 1e-6)
        ).mean().item() * 100
    else:
        precision = 0.0
        recall = 0.0

    # Same metric as your original E1/E4 scripts:
    # correct classification + exact localization / total classification + localization
    total_all = total_cls + total_loc
    correct_all = correct_cls + exact_match

    overall_acc = correct_all / total_all * 100 if total_all > 0 else 0.0

    return {
        "cls_acc": cls_acc,
        "loc_exact_acc": loc_exact_acc,
        "precision": precision,
        "recall": recall,
        "overall_acc": overall_acc,
    }


# ============================================================
# 7. MAIN TRAINING LOOP
# ============================================================

def main():
    wandb.init(
        project="SpineVQA-CL",
        name="E1a-Baseline-NewSplit",
        config={
            "experiment": "E1a",
            "architecture": "SigLIP2_CLS+BERT_CLS",
            "protocol": "train_val_test",
            "lr": cfg.LR,
            "batch_size": cfg.BATCH_SIZE,
            "epochs": cfg.EPOCHS,
            "dataset": "SpineBench",
        }
    )

    print("=" * 80)
    print("E1a — Baseline SpineVQA Training with Train/Val/Test Split")
    print("=" * 80)

    print("\nLoading processors...")
    processor = SiglipImageProcessor.from_pretrained(cfg.SIGLIP_NAME)

    tokenizer = BertTokenizer.from_pretrained(
        cfg.BERT_NAME,
        clean_up_tokenization_spaces=False
    )

    # ---------------- Datasets ----------------
    train_dataset = SpineBenchDataset(
        cfg.TRAIN_JSON,
        cfg.TRAIN_IMG_ROOT,
        processor,
        tokenizer,
        split="train"
    )

    val_dataset = SpineBenchDataset(
        cfg.VAL_JSON,
        cfg.VAL_IMG_ROOT,
        processor,
        tokenizer,
        split="val"
    )

    test_dataset = SpineBenchDataset(
        cfg.TEST_JSON,
        cfg.TEST_IMG_ROOT,
        processor,
        tokenizer,
        split="test"
    )

    # ---------------- Dataloaders ----------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )

    # ---------------- Model ----------------
    model = E1aBaselineModel().to(cfg.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ---------------- Loss / Optimizer / Scheduler ----------------
    class_weights = compute_class_weights(cfg.TRAIN_JSON)
    criterion = E1aLoss(class_weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LR,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.EPOCHS
    )

    best_val_overall = 0.0
    best_epoch = 0
    best_path = os.path.join(cfg.SAVE_DIR, "E1a_best.pth")

    print("\nStarting training...")
    print("=" * 90)
    print(
        f"{'Epoch':>6} | {'Loss':>8} | {'TrainCls':>8} | "
        f"{'ValCls':>8} | {'ValLoc':>8} | {'Pre':>7} | "
        f"{'Rec':>7} | {'ValOverall':>10} | {'LR':>10}"
    )
    print("-" * 90)

    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss, train_cls_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_metrics = evaluate(model, val_loader)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        improved = val_metrics["overall_acc"] > best_val_overall

        if improved:
            best_val_overall = val_metrics["overall_acc"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            saved = "✓"
        else:
            saved = ""

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_cls_acc": train_cls_acc,
            "val_cls_acc": val_metrics["cls_acc"],
            "val_loc_exact_acc": val_metrics["loc_exact_acc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_overall_acc": val_metrics["overall_acc"],
            "lr": current_lr,
            "best_val_overall": best_val_overall,
            "best_val_epoch": best_epoch,
        })

        print(
            f"{epoch:>6} | "
            f"{train_loss:>8.4f} | "
            f"{train_cls_acc:>7.2f}% | "
            f"{val_metrics['cls_acc']:>7.2f}% | "
            f"{val_metrics['loc_exact_acc']:>7.2f}% | "
            f"{val_metrics['precision']:>6.2f}% | "
            f"{val_metrics['recall']:>6.2f}% | "
            f"{val_metrics['overall_acc']:>9.2f}% | "
            f"{current_lr:>10.2e}  {saved}"
        )

    print("=" * 90)
    print(f"\nBest Val Overall: {best_val_overall:.2f}%")
    print(f"Best Epoch: {best_epoch}")

    # ========================================================
    # Final test evaluation only once
    # ========================================================
    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION (test.json — reported once)")
    print(f"Loading best model from epoch {best_epoch}...")
    print("=" * 80)

    model.load_state_dict(
        torch.load(
            best_path,
            map_location=cfg.DEVICE
        )
    )

    test_metrics = evaluate(model, test_loader)

    print("\nE1a FINAL TEST RESULTS:")
    print(f"  Disease Acc:  {test_metrics['cls_acc']:.2f}%")
    print(f"  Loc Acc:      {test_metrics['loc_exact_acc']:.2f}%")
    print(f"  Precision:    {test_metrics['precision']:.2f}%")
    print(f"  Recall:       {test_metrics['recall']:.2f}%")
    print(f"  Overall:      {test_metrics['overall_acc']:.2f}%")

    wandb.log({
        "test_cls_acc": test_metrics["cls_acc"],
        "test_loc_acc": test_metrics["loc_exact_acc"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_overall": test_metrics["overall_acc"],
    })

    wandb.finish()

    print(f"\nModel saved to: {best_path}")


# ============================================================
if __name__ == "__main__":
    main()
