import os
import subprocess
from setuptools import setup, find_packages, Command

class DownloadAssetsCommand(Command):
    description = "Download assets and create necessary folders."
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        repo_root = os.path.dirname(os.path.abspath(__file__))
        folders = ["data", os.path.join("src", "outputs")]

        for folder in folders:
            folder_path = os.path.join(repo_root, folder)
            os.makedirs(folder_path, exist_ok=True)
            print(f"Created folder: {folder_path}")

        print("Downloading files into data/...")

        files_to_download = [
            "babelnet_source_dict.pkl",
            "combined_lemma_embeds.pkl",
            "faiss_index_merged",
            "index_infos_merged.json",
            "country_list.pkl"
        ]

        GCLOUD = "image-cultural-evaluation"

        for file in files_to_download:
            cmd = f"gsutil cp gs://{GCLOUD}/{file} data/"
            subprocess.run(cmd, shell=True, check=True)
        print("Data download complete.")

setup(
    name="caire",
    version="0.1",
    packages=find_packages(),

    install_requires=[
        "accelerate",
        "certifi",
        "charset-normalizer",
        "einops",
        "faiss-cpu",
        "faiss-gpu",
        "filelock",
        "fsspec",
        "hf-xet",
        "huggingface-hub",
        "idna",
        "Jinja2",
        "MarkupSafe",
        "mpmath",
        "networkx",
        "numpy",
        "packaging",
        "pillow",
        "pip",
        "protobuf",
        "psutil",
        "PyYAML",
        "regex",
        "requests",
        "safetensors",
        "sentencepiece",
        "setuptools",
        "sympy",
        "tokenizers",
        "tqdm",
        "transformers",
        "triton",
        "typing_extensions",
        "urllib3",
        "wheel",
        "Wikipedia-API",
        "gsutil",
        "tabulate",
    ],
    cmdclass={
        "download_assets": DownloadAssetsCommand,
    },
)