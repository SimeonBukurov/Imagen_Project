import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CelebADataset(Dataset):
    def __init__(self, data_dir, attr_file_path, transform=None):
        self.transform = transform
        self.image_files = []
        self.labels = {} 

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory not found: {data_dir}")
        if not os.path.exists(attr_file_path):
            raise FileNotFoundError(f"Attribute file not found: {attr_file_path}")

        # 1. Load the Answer Key
        print(f"DEBUG: Parsing attributes from {attr_file_path}...")
        self._parse_attributes(attr_file_path)

        # 2. Deep Scan for Images
        print(f"DEBUG: Deep scanning '{data_dir}' for images...")
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Only add the image if we have an answer key for it
                    if file in self.labels:
                        self.image_files.append(os.path.join(root, file))
        
        print(f"DEBUG: Successfully found {len(self.image_files)} images with matching labels.")
        if len(self.image_files) == 0:
            raise ValueError("No matching images found! Check your folder paths.")

    def _parse_attributes(self, attr_file_path):
        with open(attr_file_path, 'r') as f:
            lines = f.readlines()

        # 1. Detect if this is the Kaggle CSV or the Official TXT
        if ',' in lines[0]:
            # --- KAGGLE CSV FORMAT ---
            headers = lines[0].strip().split(',')
            data_start = 1
            delimiter = ','
            offset = 0 # Headers include 'image_id' at the start
        else:
            # --- OFFICIAL TXT FORMAT ---
            headers = lines[1].strip().split()
            data_start = 2
            delimiter = None # Splits by any empty space
            offset = 1 # Headers start at attribute 1, missing 'image_id'

        # 2. Map the columns
        idx_male = headers.index("Male")
        idx_black = headers.index("Black_Hair")
        idx_blond = headers.index("Blond_Hair")
        idx_brown = headers.index("Brown_Hair")
        idx_gray = headers.index("Gray_Hair")
        idx_young = headers.index("Young")

        # 3. Read the actual answers
        for line in lines[data_start:]:
            if not line.strip(): continue # Skip blank lines
            parts = line.strip().split(delimiter)
            filename = parts[0]
            
            # Safely grab the exact number data 
            gender = 1.0 if parts[idx_male + offset] == '1' else 0.0
            black = 1.0 if parts[idx_black + offset] == '1' else 0.0
            blond = 1.0 if parts[idx_blond + offset] == '1' else 0.0
            brown = 1.0 if parts[idx_brown + offset] == '1' else 0.0
            gray = 1.0 if parts[idx_gray + offset] == '1' else 0.0
            age = 1.0 if parts[idx_young + offset] == '1' else 0.0

            # Package them for the GPU
            self.labels[filename] = torch.tensor([gender, black, blond, brown, gray, age], dtype=torch.float32)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx] 
        filename = os.path.basename(img_path) # Get just the '000001.jpg' part
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # Return the REAL answers now!
        label = self.labels[filename]
        return image, label

def get_data_loader(data_dir, attr_file_path, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = CelebADataset(data_dir, attr_file_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)