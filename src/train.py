import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def get_device():
    """
    Selects the best available device, prioritizing Apple Silicon MPS (Metal Performance Shaders)
    as per the Apple Silicon Standard.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS device for hardware acceleration.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU acceleration.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return device

def train_epoch(model, dataloader, optimizer, disease_criterion, loc_criterion, vqa_criterion, device):
    model.train()
    total_loss = 0.0
    total_disease_loss = 0.0
    total_loc_loss = 0.0
    total_vqa_loss = 0.0
    
    # Track correct predictions
    correct_disease = 0
    total_samples = 0
    
    for batch in tqdm(dataloader, desc="Training Batch"):
        images = batch["image"].to(device)
        disease_labels = batch["disease_label"].to(device)
        loc_labels = batch["loc_label"].to(device)
        
        # Move text inputs to device
        text_inputs = batch["text_inputs"]
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        
        # For VQA multiclass, map target text answers to index (dummy index for training)
        # Using disease labels as a proxy for VQA answers to simulate target multiclass mapping
        vqa_labels = disease_labels 
        
        optimizer.zero_grad()
        
        outputs = model(images, text_inputs)
        disease_logits = outputs["disease_logits"]
        loc_logits = outputs["loc_logits"]
        vqa_logits = outputs["vqa_logits"]
        
        # Compute individual multitask losses
        loss_disease = disease_criterion(disease_logits, disease_labels)
        loss_loc = loc_criterion(loc_logits, loc_labels)
        loss_vqa = vqa_criterion(vqa_logits, vqa_labels)
        
        # Combine losses
        loss = loss_disease + loss_loc + loss_vqa
        
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item() * images.size(0)
        total_disease_loss += loss_disease.item() * images.size(0)
        total_loc_loss += loss_loc.item() * images.size(0)
        total_vqa_loss += loss_vqa.item() * images.size(0)
        
        # Compute accuracy metrics
        _, preds_disease = torch.max(disease_logits, 1)
        correct_disease += torch.sum(preds_disease == disease_labels).item()
        total_samples += images.size(0)
        
    num_samples = max(total_samples, 1)
    metrics = {
        "loss": total_loss / num_samples,
        "disease_loss": total_disease_loss / num_samples,
        "loc_loss": total_loc_loss / num_samples,
        "vqa_loss": total_vqa_loss / num_samples,
        "disease_accuracy": correct_disease / num_samples
    }
    return metrics

def validate(model, dataloader, disease_criterion, loc_criterion, vqa_criterion, device):
    model.eval()
    total_loss = 0.0
    total_disease_loss = 0.0
    total_loc_loss = 0.0
    total_vqa_loss = 0.0
    
    correct_disease = 0
    total_samples = 0
    
    # Store predictions for metrics
    all_disease_preds = []
    all_disease_targets = []
    all_loc_preds = []
    all_loc_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating Batch"):
            images = batch["image"].to(device)
            disease_labels = batch["disease_label"].to(device)
            loc_labels = batch["loc_label"].to(device)
            
            text_inputs = batch["text_inputs"]
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            vqa_labels = disease_labels
            
            outputs = model(images, text_inputs)
            disease_logits = outputs["disease_logits"]
            loc_logits = outputs["loc_logits"]
            vqa_logits = outputs["vqa_logits"]
            
            loss_disease = disease_criterion(disease_logits, disease_labels)
            loss_loc = loc_criterion(loc_logits, loc_labels)
            loss_vqa = vqa_criterion(vqa_logits, vqa_labels)
            loss = loss_disease + loss_loc + loss_vqa
            
            total_loss += loss.item() * images.size(0)
            total_disease_loss += loss_disease.item() * images.size(0)
            total_loc_loss += loss_loc.item() * images.size(0)
            total_vqa_loss += loss_vqa.item() * images.size(0)
            
            _, preds_disease = torch.max(disease_logits, 1)
            correct_disease += torch.sum(preds_disease == disease_labels).item()
            total_samples += images.size(0)
            
            # Store values for evaluation metrics
            all_disease_preds.extend(preds_disease.cpu().numpy())
            all_disease_targets.extend(disease_labels.cpu().numpy())
            
            # For localization, output probability using sigmoid
            preds_loc = (torch.sigmoid(loc_logits) >= 0.5).cpu().numpy().astype(int)
            all_loc_preds.extend(preds_loc)
            all_loc_targets.extend(loc_labels.cpu().numpy())
            
    num_samples = max(total_samples, 1)
    metrics = {
        "loss": total_loss / num_samples,
        "disease_loss": total_disease_loss / num_samples,
        "loc_loss": total_loc_loss / num_samples,
        "vqa_loss": total_vqa_loss / num_samples,
        "disease_accuracy": correct_disease / num_samples,
        "disease_preds": np.array(all_disease_preds),
        "disease_targets": np.array(all_disease_targets),
        "loc_preds": np.array(all_loc_preds),
        "loc_targets": np.array(all_loc_targets)
    }
    return metrics
