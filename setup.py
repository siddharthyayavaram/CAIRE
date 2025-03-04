import os
import subprocess
import sys

repo_root = os.path.dirname(os.path.abspath(__file__))
folders = ["data", "checkpoints", os.path.join("src", "outputs")]

for folder in folders:
    folder_path = os.path.join(repo_root, folder)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Created folder: {folder_path}")

big_vision_path = os.path.join(repo_root, "big_vision")
if not os.path.exists(big_vision_path):
    print("Cloning big_vision repository...")
    subprocess.run(
        ["git", "clone", "https://github.com/google-research/big_vision", big_vision_path],
        check=True,
    )
    print("big_vision repository cloned successfully.")
else:
    print("big_vision repository already exists. Skipping clone.")

print("Downloading files into checkpoints/...")
gsutil_checkpoints = [
    "gsutil cp gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model checkpoints/",
    "gsutil cp gs://big_vision/siglip/webli_i18n_so400m_16_256_78061115.npz checkpoints/"
]

for cmd in gsutil_checkpoints:
    subprocess.run(cmd, shell=True, check=True)
print("Checkpoints download complete.")

print("Downloading files into data/...")
files_to_download = [
    "babelnet_source_dict.pkl",
    "combined_lemma_embeds.pkl",
    "faiss_index_merged",
    "index_infos_merged.json",
    "country_list.pkl"
]

for file in files_to_download:
    cmd = f"gsutil cp gs://image-cultural-evaluation/{file} data/"
    subprocess.run(cmd, shell=True, check=True)
print("Data download complete.")

conda_check = subprocess.run(["which", "conda"], capture_output=True, text=True)
if not conda_check.stdout.strip():
    print("Error: Conda is not installed. Please install Conda first.")
    sys.exit(1)

env_file = os.path.join(repo_root, "environment.yaml")
env_name = "caire"

if os.path.exists(env_file):
    print(f"Creating Conda environment '{env_name}' from environment.yaml...")
    subprocess.run(f"conda env create --file {env_file}", shell=True, check=True)
    print("Conda environment setup complete.")
else:
    print("Error: environment.yaml not found. Please provide the file.")

print("\nSetup complete! You can now activate the environment using:")
print(f"conda activate {env_name}")
