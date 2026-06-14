import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import SpineVQADataset, get_class_weights, DISEASE_CLASSES, LOCALIZATION_LEVELS
from src.model import SpineVQAModel
from src.train import get_device, train_epoch, validate
from src.eval import compute_classification_metrics, compute_localization_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="SpineVQA: Multimodal Learning for SpineBench")
    parser.add_argument("--csv-train", type=str, default=None, help="Path to training CSV file")
    parser.add_argument("--csv-val", type=str, default=None, help="Path to validation CSV file")
    parser.add_argument("--img-dir", type=str, default=None, help="Directory containing images")
    parser.add_argument("--epochs", type=str, default="5", help="Number of training epochs")
    parser.add_argument("--batch-size", type=str, default="16", help="Batch size for training")
    parser.add_argument("--lr", type=str, default="1e-4", help="Learning rate")
    parser.add_argument("--vision-backbone", type=str, default="resnet18", help="Vision backbone encoder")
    parser.add_argument("--text-backbone", type=str, default="bert-base-uncased", help="Text backbone encoder")
    parser.add_argument("--dry-run", action="store_true", help="Run with synthetic dataset for verification")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Safely convert parameter values to integers/floats
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    
    # 1. Device configuration (Apple Silicon MPS prioritized)
    device = get_device()
    
    # 2. Tokenizer setup
    # In a real environment, we would import AutoTokenizer from transformers
    tokenizer = None
    if True:
        try:
            from transformers import AutoTokenizer
            print(f"Loading tokenizer for text backbone: {args.text_backbone}...")
            tokenizer = AutoTokenizer.from_pretrained(args.text_backbone)
        except Exception as e:
            print(f"Warning: Failed to load tokenizer. Using fallback string tokenization. Error: {e}")
            
    # 3. Datasets and Dataloaders
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if args.dry_run:
        print("\n--- Running in DRY-RUN mode with Synthetic Data ---")
        train_dataset = SpineVQADataset(synthetic=True, num_samples=64, tokenizer=tokenizer)
        val_dataset = SpineVQADataset(synthetic=True, num_samples=32, tokenizer=tokenizer)
    else:
        print("\n--- Loading Real SpineBench Dataset ---")
        train_dataset = SpineVQADataset(
            csv_file=args.csv_train, img_dir=args.img_dir, transform=img_transform, tokenizer=tokenizer
        )
        val_dataset = SpineVQADataset(
            csv_file=args.csv_val, img_dir=args.img_dir, transform=img_transform, tokenizer=tokenizer
        )
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Model instantiation
    print(f"\nInitializing model (Vision: {args.vision_backbone}, Text: {args.text_backbone})...")
    model = SpineVQAModel(
        vision_backbone=args.vision_backbone,
        text_backbone=args.text_backbone,
        num_disease_classes=len(DISEASE_CLASSES),
        num_loc_classes=len(LOCALIZATION_LEVELS),
        num_vqa_answers=len(DISEASE_CLASSES) # proxy
    ).to(device)
    
    # 5. Loss criteria & Optimization
    # Apply inverse frequency weighting for disease classification to mitigate severe class imbalance
    if args.dry_run:
        disease_weights = torch.ones(len(DISEASE_CLASSES), dtype=torch.float32).to(device)
    else:
        print("Computing inverse frequency weights for disease classification...")
        disease_weights = get_class_weights(train_dataset).to(device)
        
    disease_criterion = nn.CrossEntropyLoss(weight=disease_weights)
    loc_criterion = nn.BCEWithLogitsLoss() # For multi-label classification
    vqa_criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # 6. Training Loop
    print("\n--- Starting Training Loop ---")
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            disease_criterion=disease_criterion,
            loc_criterion=loc_criterion,
            vqa_criterion=vqa_criterion,
            device=device
        )
        print(f"Train Loss: {train_metrics['loss']:.4f} | Disease Loss: {train_metrics['disease_loss']:.4f} | Loc Loss: {train_metrics['loc_loss']:.4f} | VQA Loss: {train_metrics['vqa_loss']:.4f}")
        print(f"Train Disease Acc: {train_metrics['disease_accuracy'] * 100:.2f}%")
        
        # Validation
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            disease_criterion=disease_criterion,
            loc_criterion=loc_criterion,
            vqa_criterion=vqa_criterion,
            device=device
        )
        print(f"Val Loss: {val_metrics['loss']:.4f} | Disease Loss: {val_metrics['disease_loss']:.4f} | Loc Loss: {val_metrics['loc_loss']:.4f} | VQA Loss: {val_metrics['vqa_loss']:.4f}")
        print(f"Val Disease Acc: {val_metrics['disease_accuracy'] * 100:.2f}%")
        
        # Compute detailed metrics on validation predictions
        disease_stats = compute_classification_metrics(
            targets=val_metrics["disease_targets"],
            predictions=val_metrics["disease_preds"],
            classes=DISEASE_CLASSES
        )
        loc_stats = compute_localization_metrics(
            targets=val_metrics["loc_targets"],
            predictions=val_metrics["loc_preds"],
            levels=LOCALIZATION_LEVELS
        )
        
        print(f"Val Disease F1 (Macro): {disease_stats['f1_macro'] * 100:.2f}%")
        print(f"Val Localization F1 (Macro): {loc_stats['f1_macro'] * 100:.2f}%")
        print(f"Val Localization Subset Accuracy (Exact Match): {loc_stats['exact_match_accuracy'] * 100:.2f}%")
        print(f"Val Localization Hamming Loss: {loc_stats['hamming_loss']:.4f}")

    print("\n--- Training Completed Successfully! ---")

if __name__ == "__main__":
    main()
