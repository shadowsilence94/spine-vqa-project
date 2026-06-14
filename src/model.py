import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel

class SpineVQAModel(nn.Module):
    """
    Multimodal Multitask Model for SpineVQA.
    Combines visual and textual representations to jointly predict:
    1. Spine disease category (12 classes)
    2. Spine lesion localization levels (5 classes, multi-label)
    3. Multiclass VQA answer prediction
    """
    def __init__(self, vision_backbone="resnet18", text_backbone="bert-base-uncased", 
                 num_disease_classes=12, num_loc_classes=5, num_vqa_answers=20, 
                 use_pretrained=True, embed_dim=256):
        super(SpineVQAModel, self).__init__()
        
        # 1. Vision Encoder
        self.vision_backbone_name = vision_backbone.lower()
        if self.vision_backbone_name == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if use_pretrained else None)
            self.vision_dim = resnet.fc.in_features
            self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1]) # Remove fc
        else:
            # Fallback tiny CNN for custom backbones
            self.vision_dim = 128
            self.vision_encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            
        self.vision_projection = nn.Linear(self.vision_dim, embed_dim)
        
        # 2. Text Encoder
        self.text_backbone_name = text_backbone
        try:
            # Attempt to load Hugging Face transformer
            self.text_encoder = AutoModel.from_pretrained(text_backbone)
            self.text_dim = self.text_encoder.config.hidden_size
            self.has_transformer_text = True
        except Exception as e:
            print(f"HuggingFace transformer loading failed: {e}. Using dummy embedding projection.")
            self.text_dim = 128
            self.text_encoder = nn.Embedding(30522, self.text_dim) # Vocab size of BERT
            self.has_transformer_text = False
            
        self.text_projection = nn.Linear(self.text_dim, embed_dim)
        
        # 3. Multimodal Fusion (Simple concatenation + projection)
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 4. Multitask Prediction Heads
        # Task 1: Disease classification (12 classes)
        self.disease_head = nn.Linear(embed_dim, num_disease_classes)
        
        # Task 2: Vertebral level localization (5 levels, multi-label)
        self.loc_head = nn.Linear(embed_dim, num_loc_classes)
        
        # Task 3: VQA Answer Prediction (Multiclass)
        self.vqa_head = nn.Linear(embed_dim, num_vqa_answers)

    def forward(self, images, text_inputs):
        # Extract visual features
        vis_features = self.vision_encoder(images)
        vis_features = torch.flatten(vis_features, 1) # Shape: (batch_size, vision_dim)
        vis_embed = self.vision_projection(vis_features) # Shape: (batch_size, embed_dim)
        
        # Extract textual features
        if self.has_transformer_text:
            # Hugging Face output
            # Extract input ids and attention mask
            input_ids = text_inputs.get("input_ids")
            attention_mask = text_inputs.get("attention_mask")
            if input_ids is None:
                # Fallback dummy inputs for dry-run without tokenizer
                batch_size = images.shape[0]
                input_ids = torch.zeros((batch_size, 64), dtype=torch.long, device=images.device)
                attention_mask = torch.ones((batch_size, 64), dtype=torch.long, device=images.device)
            transformer_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            # Use pooler output or mean of sequence outputs
            if hasattr(transformer_out, "pooler_output") and transformer_out.pooler_output is not None:
                text_features = transformer_out.pooler_output
            else:
                text_features = transformer_out.last_hidden_state[:, 0, :] # CLS token
        else:
            # Fallback embedding average
            dummy_input = text_inputs.get("dummy_input")
            text_features = self.text_encoder(dummy_input).mean(dim=1)
            
        text_embed = self.text_projection(text_features) # Shape: (batch_size, embed_dim)
        
        # Multimodal Fusion
        fused = torch.cat([vis_embed, text_embed], dim=1) # Shape: (batch_size, embed_dim * 2)
        fused_features = self.fusion_layer(fused) # Shape: (batch_size, embed_dim)
        
        # Predict task targets
        disease_logits = self.disease_head(fused_features)
        loc_logits = self.loc_head(fused_features)
        vqa_logits = self.vqa_head(fused_features)
        
        return {
            "disease_logits": disease_logits,
            "loc_logits": loc_logits,
            "vqa_logits": vqa_logits
        }
