# ============================================================
# E4 — LAVP-Net (Clean): Vertebral Attention + Question-Guided Fusion
# Thesis: Localization-Aware Fine-Grained Contrastive Learning
#         for Spinal Pathology Representation in Medical VQA
# Dataset: SpineBench (ACM MM 2025)
#
# Novel Components (vs E1/E2/E3):
#   1. VertebralAttention  — patch tokens → 5 level features
#   2. QuestionGuidedFusion — question attends to level features
#
# NOTE: PrototypeBank excluded for clean ablation.
#       See E5 for E4 + PrototypeBank.
#
# Architecture:
#   SigLIP2 patch tokens [B,196,768]
#       ↓
#   VertebralAttention → [B,5,768]
#       ↓
#   QuestionGuidedFusion → [B,512]
#       ↓
#   Disease Head + Location Head
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
    NUM_PATCHES = 196   # 14×14 patches

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

    LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
    DISEASE2IDX = {d: i for i, d in enumerate(DISEASES)}
    LEVEL2IDX   = {l: i for i, l in enumerate(LEVELS)}

cfg = Config()
os.makedirs(cfg.SAVE_DIR, exist_ok=True)
print(f"Device: {cfg.DEVICE}")


# ============================================================
# 1. DATASET (with paired label injection — same as E3)
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

        # Build paired lookup maps
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

        # Image
        img_path   = os.path.join(self.img_root, sample["image"])
        image      = Image.open(img_path).convert("RGB")
        img_tensor = self.processor(
            images=image,
            return_tensors="pt"
        )["pixel_values"].squeeze(0)

        # Question
        tokens = self.tokenizer(
            sample["question"],
            max_length=cfg.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids      = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        # Task & Labels
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
# 2. LAVP-NET ARCHITECTURE
# ============================================================

# ── 2a. Vertebral Attention Module ─────────────────────────
class VertebralAttention(nn.Module):
    """
    Novel Module 1: Vertebral Attention

    Uses 5 learnable vertebral level queries to attend over
    SigLIP2 patch tokens, producing level-specific features.

    Input:  patch_tokens [B, 196, 768]
    Output: level_feats  [B, 5, 768]
            attn_weights [B, 5, 196]

    Each query corresponds to one spinal level:
    Query 0 → L1/L2 feature
    Query 1 → L2/L3 feature
    Query 2 → L3/L4 feature
    Query 3 → L4/L5 feature
    Query 4 → L5/S1 feature
    """
    def __init__(self, dim=768, num_levels=5, num_heads=8):
        super().__init__()
        # Learnable level queries — trained end-to-end
        self.level_queries = nn.Parameter(
            torch.randn(num_levels, dim)
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, patch_tokens):
        """
        patch_tokens: [B, 196, 768]
        """
        B = patch_tokens.size(0)

        # Expand queries for batch
        queries = self.level_queries.unsqueeze(0).expand(B, -1, -1)
        # queries: [B, 5, 768]

        # Cross-attention: queries attend to patch tokens
        level_feats, attn_weights = self.attn(
            query=queries,
            key=patch_tokens,
            value=patch_tokens
        )
        # level_feats:  [B, 5, 768]
        # attn_weights: [B, 5, 196]

        level_feats = self.norm(level_feats)
        return level_feats, attn_weights


# ── 2b. Question-Guided Fusion ─────────────────────────────
class QuestionGuidedFusion(nn.Module):
    """
    Novel Module 2: Question-Guided Cross-Attention Fusion

    Question feature attends over level-specific features,
    selecting the most relevant vertebral level for the query.

    Classification question → attends to disease-relevant level
    Localization question   → attends to spatial level features

    Input:
        level_feats: [B, 5, 768]  — vertebral level features
        q_feat:      [B, 768]     — question feature
    Output:
        fused:       [B, 512]     — fused representation
    """
    def __init__(self, img_dim=768, q_dim=768, out_dim=512,
                 num_heads=8):
        super().__init__()

        # Project question to same dim as level features
        self.q_proj = nn.Linear(q_dim, img_dim)

        # Cross-attention: question attends to levels
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=img_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        self.norm1 = nn.LayerNorm(img_dim)

        # Final projection to output dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(img_dim + q_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, level_feats, q_feat):
        """
        level_feats: [B, 5, 768]
        q_feat:      [B, 768]
        """
        # Project question
        q_proj = self.q_proj(q_feat)          # [B, 768]
        q_query = q_proj.unsqueeze(1)          # [B, 1, 768]

        # Question attends to vertebral level features
        attn_out, _ = self.cross_attn(
            query=q_query,
            key=level_feats,
            value=level_feats
        )
        # attn_out: [B, 1, 768]
        attn_out = self.norm1(attn_out).squeeze(1)
        # attn_out: [B, 768]

        # Concatenate with question and project
        fused = torch.cat([attn_out, q_feat], dim=-1)
        # fused: [B, 1536]
        fused = self.fusion_proj(fused)
        # fused: [B, 512]

        return fused


# ── 2c. Disease-Location Prototype Bank ────────────────────
class DiseaseLocationPrototypeBank(nn.Module):
    """
    Novel Module 3: Disease-Location Prototype Bank

    Creates 12 × 5 = 60 learnable prototypes, one for each
    (disease, spinal level) combination.

    During training: contrastive loss pulls features toward
    their correct prototype and pushes away from wrong ones.

    Prototypes:
        P[0][0] = Subarticular Stenosis at L1/L2
        P[0][1] = Subarticular Stenosis at L2/L3
        ...
        P[11][4] = Vertebral collapse at L5/S1
    """
    def __init__(self, num_disease=12, num_levels=5,
                 feat_dim=512, temperature=0.07):
        super().__init__()
        self.num_disease = num_disease
        self.num_levels  = num_levels
        self.temperature = temperature

        # Learnable prototypes [12, 5, 512]
        self.prototypes = nn.Parameter(
            F.normalize(
                torch.randn(num_disease, num_levels, feat_dim),
                dim=-1
            )
        )

    def forward(self, features, disease_labels, loc_labels):
        """
        Args:
            features:       [B, 512]
            disease_labels: [B]      (-1 = localization only)
            loc_labels:     [B, 5]   multi-hot

        Returns:
            prototype contrastive loss scalar
        """
        # Select paired samples
        has_disease  = disease_labels >= 0
        has_location = loc_labels.sum(dim=1) > 0
        paired_mask  = has_disease & has_location

        if paired_mask.sum() < 2:
            return torch.tensor(0.0,
                device=features.device, requires_grad=True)

        feat  = features[paired_mask]
        d_lb  = disease_labels[paired_mask]
        l_lb  = loc_labels[paired_mask]

        # Normalize features
        feat = F.normalize(feat, dim=1)

        # Get target prototype for each sample
        # Use dominant level (argmax of loc_label)
        level_idx = l_lb.argmax(dim=1)  # [N]

        # Gather target prototypes [N, 512]
        target_proto = self.prototypes[d_lb, level_idx]
        target_proto = F.normalize(target_proto, dim=1)

        # Positive similarity: feature vs target prototype
        pos_sim = (feat * target_proto).sum(dim=1) / self.temperature
        # [N]

        # Negative similarities: feature vs ALL 60 prototypes
        all_proto = F.normalize(
            self.prototypes.view(-1, feat.size(-1)), dim=1
        )
        # all_proto: [60, 512]

        all_sim = feat @ all_proto.T / self.temperature
        # all_sim: [N, 60]

        # Get target prototype index
        target_idx = d_lb * self.num_levels + level_idx
        # [N]

        # Contrastive loss
        loss = F.cross_entropy(all_sim, target_idx)

        return loss


# ── 2d. LAVP-Net Main Model ────────────────────────────────
class LAVPNet(nn.Module):
    """
    LAVP-Net: Localization-Aware Vertebral Prototype Network

    Full architecture combining all novel components:
    1. SigLIP2 → patch tokens (not just CLS)
    2. VertebralAttention → 5 level features
    3. QuestionGuidedFusion → question-aware representation
    4. DiseaseLocationPrototypeBank → prototype contrastive

    SigLIP2 remains FROZEN for fair ablation with E1-E3.
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

        # ── Novel Module 1: Vertebral Attention ────────
        self.vertebral_attn = VertebralAttention(
            dim=cfg.IMG_DIM,
            num_levels=cfg.NUM_LEVELS,
            num_heads=8
        )

        # ── Novel Module 2: Question-Guided Fusion ─────
        self.qg_fusion = QuestionGuidedFusion(
            img_dim=cfg.IMG_DIM,
            q_dim=cfg.Q_DIM,
            out_dim=cfg.HIDDEN_DIM,
            num_heads=8
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
            fused:          [B, 512]  for contrastive loss
            attn_weights:   [B, 5, 196]  vertebral attention maps
        """
        # ── SigLIP2: Extract ALL patch tokens ─────────
        with torch.no_grad():
            vision_out = self.siglip2.vision_model(
                pixel_values=images.half()
            )
            # ALL tokens: [B, 197, 768]
            # index 0 = CLS, index 1: = patches
            all_tokens   = vision_out.last_hidden_state.float()
            patch_tokens = all_tokens[:, 1:, :]
            # patch_tokens: [B, 196, 768]

        # ── BERT: Question Encoding ────────────────────
        q_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        q_feat = q_out.last_hidden_state[:, 0, :]
        # q_feat: [B, 768]

        # ── Novel 1: Vertebral Attention ───────────────
        level_feats, attn_weights = self.vertebral_attn(patch_tokens)
        # level_feats:  [B, 5, 768]
        # attn_weights: [B, 5, 196]

        # ── Novel 2: Question-Guided Fusion ───────────
        fused = self.qg_fusion(level_feats, q_feat)
        # fused: [B, 512]

        # ── Heads ──────────────────────────────────────
        disease_logits = self.disease_head(fused)
        loc_logits     = self.loc_head(fused)

        return disease_logits, loc_logits, fused, attn_weights


# ============================================================
# 3. LOSS FUNCTIONS
# ============================================================

def compute_class_weights(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    counts = torch.zeros(cfg.NUM_DISEASE)
    for sample in data:
        if sample["task"] == "spine_disease_classification":
            ans = sample["answers"]
            if isinstance(ans, list): ans = ans[0]
            idx = cfg.DISEASE2IDX.get(ans, -1)
            if idx >= 0:
                counts[idx] += 1
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * cfg.NUM_DISEASE
    print("Class weights computed:")
    for i, (d, w) in enumerate(zip(cfg.DISEASES, weights)):
        print(f"  {d[:30]:30s}: {w:.3f}")
    return weights


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
            L_d = self.disease_loss(
                disease_logits[cls_mask],
                disease_labels[cls_mask]
            )
            total_loss = total_loss + L_d
        else:
            L_d = torch.tensor(0.0)

        loc_mask = torch.tensor(
            [t == "spine_lesion_localization" for t in tasks],
            dtype=torch.bool
        ).to(cfg.DEVICE)
        if loc_mask.any():
            L_l = self.loc_loss(
                loc_logits[loc_mask],
                loc_labels[loc_mask]
            )
            total_loss = total_loss + L_l
        else:
            L_l = torch.tensor(0.0)

        return total_loss, L_d, L_l


# ============================================================
# 4. TRAIN ONE EPOCH
# ============================================================
def train_epoch(model, loader, optimizer, task_criterion):
    model.train()

    total_loss  = 0.0
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

        # Forward — LAVP-Net returns 4 values
        disease_logits, loc_logits, fused, _ = model(
            images, input_ids, attention_mask
        )

        # Task loss only (no contrastive in E4 clean)
        loss, L_d, L_l = task_criterion(
            disease_logits, loc_logits,
            disease_labels, loc_labels, tasks
        )

        # Backward
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

            # LAVP-Net returns 4 values
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
        name="E4-LAVPNet-Clean",
        config={
            "lr":                  cfg.LR,
            "batch_size":          cfg.BATCH_SIZE,
            "epochs":              cfg.EPOCHS,
            "model":               "LAVP-Net (Clean)",
            "novel_modules":       "VertebralAttn+QGFusion",
            "prototype_bank":      False,
            "siglip2_frozen":      True,
            "dataset":             "SpineBench"
        }
    )

    print("=" * 60)
    print("E4 — LAVP-Net (Clean) Training")
    print("VertebralAttention + QuestionGuidedFusion")
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

    # LAVP-Net model
    model = LAVPNet().to(cfg.DEVICE)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"\nNovel modules:")
    print(f"  VertebralAttention:  {sum(p.numel() for p in model.vertebral_attn.parameters()):,} params")
    print(f"  QuestionGuidedFusion:{sum(p.numel() for p in model.qg_fusion.parameters()):,} params")

    # Loss functions
    class_weights  = compute_class_weights(cfg.TRAIN_JSON)
    task_criterion = TaskLoss(class_weights)

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LR,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS
    )

    # Training loop
    best_overall = 0.0
    print("\nStarting training...")
    print("=" * 80)
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Cls Acc':>8} | "
          f"{'Loc Acc':>8} | {'Pre':>7} | {'Rec':>7} | "
          f"{'Overall':>8} | {'LR':>10}")
    print("-" * 80)

    for epoch in range(1, cfg.EPOCHS + 1):

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, task_criterion
        )

        metrics    = evaluate(model, test_loader)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

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

        if metrics["overall_acc"] > best_overall:
            best_overall = metrics["overall_acc"]
            torch.save(
                model.state_dict(),
                os.path.join(cfg.SAVE_DIR, "E4_best.pth")
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
    print(f"E3 Baseline:           34.40%")
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
    print(f"\nModel saved to: {cfg.SAVE_DIR}/E4_best.pth")


# ============================================================
if __name__ == "__main__":
    main()
