import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# 12 Disease Categories defined in SpineBench
DISEASE_CLASSES = [
    "Subarticular Stenosis",
    "Foraminal Stenosis",
    "Healthy",
    "Osteophytes",
    "Spinal Canal Stenosis",
    "Cervical Lordosis",
    "Straight Cervical Vertebrae",
    "Sigmoid Cervical Vertebrae",
    "Cervical Kyphosis",
    "Disc Space Narrowing",
    "Spondylolisthesis",
    "Vertebral Collapse"
]

# 5 Lumbar Vertebral Levels for Localization
LOCALIZATION_LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

class SpineVQADataset(Dataset):
    """
    Dataset class for SpineBench Visual Question Answering & Lesion Localization.
    Supports real CSV/image loading or synthetic mode for dry runs.
    """
    def __init__(self, csv_file=None, img_dir=None, transform=None, tokenizer=None, max_length=64, synthetic=False, num_samples=100):
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.synthetic = synthetic
        
        if self.synthetic:
            self.samples = self._generate_synthetic_samples(num_samples)
        else:
            self.samples = self._load_dataset()
            
    def _generate_synthetic_samples(self, num_samples):
        samples = []
        templates = [
            ("What spinal disease is present in this image?", "disease"),
            ("Which vertebral level is affected?", "location"),
            ("Is there any subarticular stenosis here?", "boolean")
        ]
        
        np.random.seed(42)
        for idx in range(num_samples):
            q_type = np.random.choice(["disease", "location", "boolean"])
            if q_type == "disease":
                question = np.random.choice([t[0] for t in templates if t[1] == "disease"])
                answer = np.random.choice(DISEASE_CLASSES)
            elif q_type == "location":
                question = np.random.choice([t[0] for t in templates if t[1] == "location"])
                # e.g., L4/L5 is affected
                answer = np.random.choice(LOCALIZATION_LEVELS)
            else:
                question = np.random.choice([t[0] for t in templates if t[1] == "boolean"])
                answer = np.random.choice(["Yes", "No"])
                
            disease_idx = np.random.randint(0, len(DISEASE_CLASSES))
            # 5-dimensional multi-label vector for vertebral levels
            loc_label = np.random.randint(0, 2, size=(5,)).astype(np.float32)
            
            samples.append({
                "image_path": f"synthetic_{idx}.png",
                "question": question,
                "answer": answer,
                "disease_label": disease_idx,
                "loc_label": loc_label
            })
        return samples
        
    def _load_dataset(self):
        # Placeholder for loading actual SpineBench CSV files
        # Returns list of dicts: [{"image_path": ..., "question": ..., "answer": ..., "disease_label": ..., "loc_label": ...}]
        if not self.csv_file or not os.path.exists(self.csv_file):
            print(f"Warning: CSV file '{self.csv_file}' not found. Falling back to synthetic mode.")
            self.synthetic = True
            return self._generate_synthetic_samples(100)
            
        # In a real setup, parse CSV file:
        import pandas as pd
        df = pd.read_csv(self.csv_file)
        samples = []
        for _, row in df.iterrows():
            # Parse localized labels from row (e.g. comma-separated levels)
            loc = np.zeros(5, dtype=np.float32)
            if "levels" in row and isinstance(row["levels"], str):
                for lvl in row["levels"].split(","):
                    lvl = lvl.strip()
                    if lvl in LOCALIZATION_LEVELS:
                        loc[LOCALIZATION_LEVELS.index(lvl)] = 1.0
            
            samples.append({
                "image_path": os.path.join(self.img_dir, row["image_name"]) if self.img_dir else row["image_name"],
                "question": row["question"],
                "answer": row.get("answer", ""),
                "disease_label": int(row.get("disease_class_idx", 0)),
                "loc_label": loc
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load visual input
        if self.synthetic:
            # Generate synthetic 3-channel image
            image = torch.randn(3, 224, 224)
        else:
            img_path = sample["image_path"]
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                # Fallback to dummy tensor if image loading fails
                image = Image.new("RGB", (224, 224), color="black")
                
            if self.transform:
                image = self.transform(image)
            else:
                from torchvision.transforms import ToTensor, Resize, Compose
                default_transform = Compose([Resize((224, 224)), ToTensor()])
                image = default_transform(image)
                
        # Tokenize question
        question = sample["question"]
        if self.tokenizer:
            text_inputs = self.tokenizer(
                question,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            # Remove batch dimension added by return_tensors="pt"
            text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
        else:
            # Simple word count encoding as fallback
            text_inputs = {"dummy_input": torch.zeros(self.max_length, dtype=torch.long)}
            
        # Parse answers
        # For simplicity, answers are mapped to index or kept as string
        # In a closed-vocabulary setup, answer matches disease index or location index
        disease_label = torch.tensor(sample["disease_label"], dtype=torch.long)
        loc_label = torch.tensor(sample["loc_label"], dtype=torch.float32)
        
        return {
            "image": image,
            "text_inputs": text_inputs,
            "question": question,
            "answer": sample["answer"],
            "disease_label": disease_label,
            "loc_label": loc_label
        }

def get_class_weights(dataset):
    """
    Computes inverse frequency class weights to address severe class imbalance.
    """
    labels = [sample["disease_label"] for sample in dataset.samples]
    class_counts = np.bincount(labels, minlength=len(DISEASE_CLASSES))
    total_samples = len(labels)
    
    # Compute inverse frequency: total_samples / (num_classes * count)
    weights = np.zeros_like(class_counts, dtype=np.float32)
    for i, count in enumerate(class_counts):
        if count > 0:
            weights[i] = total_samples / (len(DISEASE_CLASSES) * count)
        else:
            weights[i] = 1.0  # Fallback for classes with no samples
            
    # Normalize weights to sum to num_classes
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)
