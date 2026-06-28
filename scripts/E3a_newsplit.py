import os
import json
import math
import random
import warnings
import wandb
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoImageProcessor, AutoModel, BertTokenizer, BertModel

warnings.filterwarnings("ignore")


# ============================================================
# Config
# ============================================================
class Config:
    DATA_ROOT = "/home/dsia-st125985/SpineVQA/data/SpineBench"

    TRAIN_JSON = f"{DATA_ROOT}/all/train_split.json"
    VAL_JSON = f"{DATA_ROOT}/all/val_split.json"
    TEST_JSON = f"{DATA_ROOT}/evaluation/test.json"

    TRAIN_IMG_ROOT = f"{DATA_ROOT}/all"
    VAL_IMG_ROOT = f"{DATA_ROOT}/all"
    TEST_IMG_ROOT = f"{DATA_ROOT}/evaluation"

    SAVE_DIR = "/home/dsia-st125985/SpineVQA/models"
    os.makedirs(SAVE_DIR, exist_ok=True)

    VISION_MODEL = "google/siglip2-base-patch16-224"
    TEXT_MODEL = "bert-base-uncased"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    SEED = 42
    EPOCHS = 20
    BATCH_SIZE = 16
    NUM_WORKERS = 4

    LR = 1e-4
    WEIGHT_DECAY = 1e-4

    IMAGE_DIM = 768
    TEXT_DIM = 768
    HIDDEN_DIM = 512
    PROJ_DIM = 512

    DROPOUT = 0.3

    CONTRASTIVE_TEMP = 0.07
    CONTRASTIVE_LAMBDA = 0.30

    NUM_DISEASES = 12
    NUM_LEVELS = 5


cfg = Config()


DISEASE_CLASSES = [
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

LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

DISEASE_TO_ID = {d.lower(): i for i, d in enumerate(DISEASE_CLASSES)}
LEVEL_TO_ID = {l: i for i, l in enumerate(LEVELS)}


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["data", "samples", "annotations"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(f"Unsupported JSON format: {path}")


def get_answer(sample):
    return sample.get("answer", sample.get("answers", ""))


def image_key(sample):
    img = sample.get("image", sample.get("image_path", ""))
    return os.path.basename(img)


def is_classification_task(sample):
    task = sample.get("task", sample.get("task_type", "")).lower()
    return "classification" in task or "diagnosis" in task or "disease" in task


def is_localization_task(sample):
    task = sample.get("task", sample.get("task_type", "")).lower()
    return "localization" in task or "lesion" in task or "location" in task


def parse_disease(answer):
    ans = str(answer).lower()

    for disease in DISEASE_CLASSES:
        if disease.lower() in ans:
            return DISEASE_TO_ID[disease.lower()]

    return -1


def parse_levels(answer):
    ans = str(answer)
    target = torch.zeros(len(LEVELS), dtype=torch.float32)

    for level in LEVELS:
        if level in ans:
            target[LEVEL_TO_ID[level]] = 1.0

    return target


def level_bitmask(level_tensor):
    """
    Convert multi-hot level vector into one integer.
    Example:
    [0,0,0,1,1] -> bitmask.
    """
    mask = 0
    for i, v in enumerate(level_tensor):
        if float(v) > 0.5:
            mask += 2 ** i
    return mask


def resolve_image_path(img_root, sample):
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

    raise FileNotFoundError(f"Image not found. Tried: {candidates[:5]} ...")


# ============================================================
# Dataset
# ============================================================
class SpineBenchDataset(Dataset):
    def __init__(self, json_path, img_root, processor, tokenizer, split="train"):
        self.samples = load_json(json_path)
        self.img_root = img_root
        self.processor = processor
        self.tokenizer = tokenizer
        self.split = split

        self.image_to_disease = {}
        self.image_to_levels = {}

        self._build_image_label_maps()

    def _build_image_label_maps(self):
        """
        E3a contrastive needs same disease + same spinal level.
        The dataset has separate classification and localization VQA pairs.
        So we build image-level maps:
        image -> disease
        image -> lesion-level multi-hot
        """
        for sample in self.samples:
            key = image_key(sample)
            ans = get_answer(sample)

            if is_classification_task(sample):
                d = parse_disease(ans)
                if d >= 0:
                    self.image_to_disease[key] = d

            if is_localization_task(sample):
                lv = parse_levels(ans)
                if lv.sum() > 0:
                    self.image_to_levels[key] = lv

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_path = resolve_image_path(self.img_root, sample)
        image = Image.open(img_path).convert("RGB")

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        )["pixel_values"].squeeze(0)

        question = sample.get("question", sample.get("query", ""))
        encoded = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        ans = get_answer(sample)
        key = image_key(sample)

        task_type = 0 if is_classification_task(sample) else 1

        disease_label = torch.tensor(-1, dtype=torch.long)
        loc_target = torch.zeros(cfg.NUM_LEVELS, dtype=torch.float32)

        if task_type == 0:
            disease_label = torch.tensor(parse_disease(ans), dtype=torch.long)

            if key in self.image_to_levels:
                loc_target = self.image_to_levels[key].float()

        else:
            loc_target = parse_levels(ans)

            if key in self.image_to_disease:
                disease_label = torch.tensor(
                    self.image_to_disease[key],
                    dtype=torch.long
                )

        # E3a contrastive label:
        # positive pairs require same disease AND same exact level pattern
        con_label = -1

        d_for_con = int(disease_label.item())
        lv_for_con = loc_target

        if d_for_con >= 0 and lv_for_con.sum() > 0:
            lv_mask = level_bitmask(lv_for_con)
            con_label = d_for_con * 100 + lv_mask

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "task_type": torch.tensor(task_type, dtype=torch.long),
            "disease_label": disease_label,
            "loc_target": loc_target,
            "contrastive_label": torch.tensor(con_label, dtype=torch.long),
        }


# ============================================================
# Class weights
# ============================================================
def compute_class_weights(json_path):
    samples = load_json(json_path)

    counts = Counter()
    for sample in samples:
        if is_classification_task(sample):
            d = parse_disease(get_answer(sample))
            if d >= 0:
                counts[d] += 1

    weights = torch.ones(cfg.NUM_DISEASES, dtype=torch.float32)

    total = sum(counts.values())
    for i in range(cfg.NUM_DISEASES):
        if counts[i] > 0:
            weights[i] = total / (cfg.NUM_DISEASES * counts[i])
        else:
            weights[i] = 1.0

    # Normalize for stability
    weights = weights / weights.mean()

    print("\nClass counts:")
    for i, name in enumerate(DISEASE_CLASSES):
        print(f"  {name:35s}: {counts[i]:6d} | weight={weights[i]:.3f}")

    return weights


# ============================================================
# Model
# ============================================================
class E3aLocalizationContrastiveModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.vision_model = AutoModel.from_pretrained(cfg.VISION_MODEL)
        self.text_model = BertModel.from_pretrained(cfg.TEXT_MODEL)

        # E3a follows E1a/E2a style: keep SigLIP2 frozen
        for p in self.vision_model.parameters():
            p.requires_grad = False

        self.image_proj = nn.Sequential(
            nn.Linear(cfg.IMAGE_DIM, cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(cfg.TEXT_DIM, cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
        )

        self.fusion = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM * 2, cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
        )

        self.proj_head = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM, cfg.PROJ_DIM),
            nn.ReLU(),
            nn.Linear(cfg.PROJ_DIM, cfg.PROJ_DIM),
        )

        self.disease_head = nn.Linear(cfg.HIDDEN_DIM, cfg.NUM_DISEASES)
        self.loc_head = nn.Linear(cfg.HIDDEN_DIM, cfg.NUM_LEVELS)

    def encode_image_cls(self, pixel_values):
        with torch.no_grad():
            vision_outputs = self.vision_model.vision_model(
                pixel_values=pixel_values
            )

            if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
                image_feat = vision_outputs.pooler_output
            else:
                image_feat = vision_outputs.last_hidden_state[:, 0, :]

        return image_feat

    def forward(self, pixel_values, input_ids, attention_mask):
        image_feat = self.encode_image_cls(pixel_values)
        image_feat = self.image_proj(image_feat)

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_feat = text_outputs.last_hidden_state[:, 0, :]
        text_feat = self.text_proj(text_feat)

        fused = torch.cat([image_feat, text_feat], dim=1)
        fused = self.fusion(fused)

        z = self.proj_head(fused)
        z = F.normalize(z, dim=1)

        disease_logits = self.disease_head(fused)
        loc_logits = self.loc_head(fused)

        return disease_logits, loc_logits, z


# ============================================================
# Losses
# ============================================================
class TaskLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, disease_logits, loc_logits, batch):
        task_type = batch["task_type"]
        disease_label = batch["disease_label"]
        loc_target = batch["loc_target"]

        total_loss = 0.0
        used = 0

        cls_mask = task_type == 0
        if cls_mask.any():
            cls_loss = self.ce(
                disease_logits[cls_mask],
                disease_label[cls_mask]
            )
            total_loss = total_loss + cls_loss
            used += 1

        loc_mask = task_type == 1
        if loc_mask.any():
            loc_loss = self.bce(
                loc_logits[loc_mask],
                loc_target[loc_mask]
            )
            total_loss = total_loss + loc_loss
            used += 1

        if used == 0:
            return torch.tensor(0.0, device=disease_logits.device)

        return total_loss


class LocalizationAwareSupConLoss(nn.Module):
    """
    E3a contrastive loss.

    Positive pairs:
        same disease AND same exact spinal-level pattern.

    contrastive_label = disease_id * 100 + level_bitmask

    If an anchor has no positive pair in the batch, it is ignored.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z, labels):
        device = z.device

        valid = labels >= 0
        if valid.sum() < 2:
            return torch.tensor(0.0, device=device)

        z = z[valid]
        labels = labels[valid]

        n = z.size(0)

        sim = torch.matmul(z, z.T) / self.temperature

        self_mask = torch.eye(n, dtype=torch.bool, device=device)
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = pos_mask & (~self_mask)

        anchor_has_pos = pos_mask.sum(dim=1) > 0
        if anchor_has_pos.sum() == 0:
            return torch.tensor(0.0, device=device)

        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        sim = sim.masked_fill(self_mask, -1e9)

        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1)
        pos_count = pos_mask.sum(dim=1).clamp(min=1)

        loss_per_anchor = -pos_log_prob / pos_count
        loss = loss_per_anchor[anchor_has_pos].mean()

        return loss


# ============================================================
# Train / Evaluate
# ============================================================
def move_batch_to_device(batch):
    return {
        k: v.to(cfg.DEVICE) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def train_one_epoch(model, train_loader, optimizer, task_loss_fn, con_loss_fn, epoch):
    model.train()

    total_loss = 0.0
    total_task_loss = 0.0
    total_con_loss = 0.0

    correct_cls = 0
    total_cls = 0

    for step, batch in enumerate(train_loader):
        batch = move_batch_to_device(batch)

        disease_logits, loc_logits, z = model(
            batch["pixel_values"],
            batch["input_ids"],
            batch["attention_mask"]
        )

        task_loss = task_loss_fn(disease_logits, loc_logits, batch)
        con_loss = con_loss_fn(z, batch["contrastive_label"])

        loss = task_loss + cfg.CONTRASTIVE_LAMBDA * con_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_task_loss += task_loss.item()
        total_con_loss += con_loss.item()

        cls_mask = batch["task_type"] == 0
        if cls_mask.any():
            preds = disease_logits[cls_mask].argmax(dim=1)
            labels = batch["disease_label"][cls_mask]
            correct_cls += (preds == labels).sum().item()
            total_cls += labels.numel()

        if step % 100 == 0:
            print(
                f"Epoch {epoch:02d} | Step {step:04d}/{len(train_loader)} "
                f"| Loss {loss.item():.4f} "
                f"| Task {task_loss.item():.4f} "
                f"| Con {con_loss.item():.4f}"
            )

    train_cls_acc = 100.0 * correct_cls / max(total_cls, 1)

    return {
        "loss": total_loss / len(train_loader),
        "task_loss": total_task_loss / len(train_loader),
        "con_loss": total_con_loss / len(train_loader),
        "train_cls_acc": train_cls_acc,
    }


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    correct_cls = 0
    total_cls = 0

    exact_loc = 0
    total_loc = 0

    precision_scores = []
    recall_scores = []

    for batch in loader:
        batch = move_batch_to_device(batch)

        disease_logits, loc_logits, _ = model(
            batch["pixel_values"],
            batch["input_ids"],
            batch["attention_mask"]
        )

        task_type = batch["task_type"]

        cls_mask = task_type == 0
        if cls_mask.any():
            preds = disease_logits[cls_mask].argmax(dim=1)
            labels = batch["disease_label"][cls_mask]
            valid = labels >= 0

            if valid.any():
                correct_cls += (preds[valid] == labels[valid]).sum().item()
                total_cls += valid.sum().item()

        loc_mask = task_type == 1
        if loc_mask.any():
            probs = torch.sigmoid(loc_logits[loc_mask])
            preds = (probs >= 0.5).float()
            targets = batch["loc_target"][loc_mask]

            matches = (preds == targets).all(dim=1)
            exact_loc += matches.sum().item()
            total_loc += targets.size(0)

            tp = (preds * targets).sum(dim=1)
            pred_sum = preds.sum(dim=1)
            gt_sum = targets.sum(dim=1)

            precision = torch.where(
                pred_sum > 0,
                tp / pred_sum.clamp(min=1),
                torch.zeros_like(tp)
            )

            recall = torch.where(
                gt_sum > 0,
                tp / gt_sum.clamp(min=1),
                torch.zeros_like(tp)
            )

            precision_scores.extend(precision.cpu().tolist())
            recall_scores.extend(recall.cpu().tolist())

    cls_acc = 100.0 * correct_cls / max(total_cls, 1)
    loc_exact_acc = 100.0 * exact_loc / max(total_loc, 1)

    precision = 100.0 * np.mean(precision_scores) if precision_scores else 0.0
    recall = 100.0 * np.mean(recall_scores) if recall_scores else 0.0

    overall_acc = 100.0 * (correct_cls + exact_loc) / max(total_cls + total_loc, 1)

    return {
        "cls_acc": cls_acc,
        "loc_exact_acc": loc_exact_acc,
        "precision": precision,
        "recall": recall,
        "overall_acc": overall_acc,
    }


# ============================================================
# Main
# ============================================================
def main():
    set_seed(cfg.SEED)

    print("=" * 80)
    print("E3a: Localization-Aware Contrastive Learning")
    print("Positive pair = same disease AND same spinal level pattern")
    print("=" * 80)
    print(f"Device: {cfg.DEVICE}")

    wandb.init(
        project="SpineVQA-CL",
        name="E3a-LocalizationContrastive-NewSplit",
        config={
            "experiment": "E3a",
            "vision_model": cfg.VISION_MODEL,
            "text_model": cfg.TEXT_MODEL,
            "epochs": cfg.EPOCHS,
            "batch_size": cfg.BATCH_SIZE,
            "lr": cfg.LR,
            "weight_decay": cfg.WEIGHT_DECAY,
            "contrastive_temp": cfg.CONTRASTIVE_TEMP,
            "contrastive_lambda": cfg.CONTRASTIVE_LAMBDA,
            "positive_pair": "same disease + same spinal level pattern",
            "train_json": cfg.TRAIN_JSON,
            "val_json": cfg.VAL_JSON,
            "test_json": cfg.TEST_JSON,
        }
    )

    processor = AutoImageProcessor.from_pretrained(cfg.VISION_MODEL)
    tokenizer = BertTokenizer.from_pretrained(cfg.TEXT_MODEL)

    print("\nLoading datasets...")
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

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples:   {len(val_dataset):,}")
    print(f"Test samples:  {len(test_dataset):,}")

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

    class_weights = compute_class_weights(cfg.TRAIN_JSON).to(cfg.DEVICE)

    model = E3aLocalizationContrastiveModel().to(cfg.DEVICE)

    task_loss_fn = TaskLoss(class_weights)
    con_loss_fn = LocalizationAwareSupConLoss(
        temperature=cfg.CONTRASTIVE_TEMP
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY
    )

    best_val_overall = 0.0
    best_epoch = 0
    best_path = os.path.join(cfg.SAVE_DIR, "E3a_best.pth")

    print("\nTraining...")
    print("=" * 80)
    print(
        f"{'Epoch':>5} | {'Loss':>8} | {'Task':>8} | {'Con':>8} | "
        f"{'TrainCls':>8} | {'ValCls':>8} | {'ValLoc':>8} | "
        f"{'ValOverall':>10} | Save"
    )
    print("-" * 80)

    for epoch in range(1, cfg.EPOCHS + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            task_loss_fn,
            con_loss_fn,
            epoch
        )

        val_metrics = evaluate(model, val_loader)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_task_loss": train_stats["task_loss"],
            "train_contrastive_loss": train_stats["con_loss"],
            "train_cls_acc": train_stats["train_cls_acc"],
            "val_cls_acc": val_metrics["cls_acc"],
            "val_loc_exact_acc": val_metrics["loc_exact_acc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_overall_acc": val_metrics["overall_acc"],
        })

        if val_metrics["overall_acc"] > best_val_overall:
            best_val_overall = val_metrics["overall_acc"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            saved = "✓"
        else:
            saved = ""

        print(
            f"{epoch:5d} | "
            f"{train_stats['loss']:8.4f} | "
            f"{train_stats['task_loss']:8.4f} | "
            f"{train_stats['con_loss']:8.4f} | "
            f"{train_stats['train_cls_acc']:8.2f} | "
            f"{val_metrics['cls_acc']:8.2f} | "
            f"{val_metrics['loc_exact_acc']:8.2f} | "
            f"{val_metrics['overall_acc']:10.2f} | {saved}"
        )

    print("=" * 80)
    print(f"\nBest Val Overall: {best_val_overall:.2f}%")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best checkpoint: {best_path}")

    print("\nFINAL TEST EVALUATION")
    print("Loading best validation checkpoint...")
    model.load_state_dict(
        torch.load(best_path, map_location=cfg.DEVICE)
    )

    test_metrics = evaluate(model, test_loader)

    print("\nE3a FINAL TEST RESULTS:")
    print(f"  Disease Acc:  {test_metrics['cls_acc']:.2f}%")
    print(f"  Loc Acc:      {test_metrics['loc_exact_acc']:.2f}%")
    print(f"  Precision:    {test_metrics['precision']:.2f}%")
    print(f"  Recall:       {test_metrics['recall']:.2f}%")
    print(f"  Overall:      {test_metrics['overall_acc']:.2f}%")

    wandb.log({
        "best_epoch": best_epoch,
        "best_val_overall": best_val_overall,
        "test_cls_acc": test_metrics["cls_acc"],
        "test_loc_exact_acc": test_metrics["loc_exact_acc"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_overall_acc": test_metrics["overall_acc"],
    })

    wandb.finish()



    print("\nComparison:")
    print("  E1a baseline overall: 33.27%")
    print("  E4b current best:     58.32%")
    print("  E3a tests whether strict disease+level contrastive loss helps.")


if __name__ == "__main__":
    main()
