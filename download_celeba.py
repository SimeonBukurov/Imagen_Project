import kagglehub
import shutil
import os

print("Downloading CelebA from Kaggle... (This is over 1GB, please be patient)")
cache_path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
print(f"Downloaded to system cache: {cache_path}")

# Pointing to the exact folder you just created
project_data_path = r"D:\Imagen Classes UPV\Project\data\raw"

print(f"Transferring 200,000+ files to: {project_data_path}...")
shutil.copytree(cache_path, project_data_path, dirs_exist_ok=True)

print("✅ Success! The dataset is ready for training.")