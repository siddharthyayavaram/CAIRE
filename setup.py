import os
import subprocess
from setuptools import setup, find_packages, Command

class DownloadAssetsCommand(Command):
    description = "Download assets, clone repositories, and create necessary folders."
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
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

        print("Downloading models into checkpoints/...")
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
        "absl-py",
        "accelerate",
        "aqtp",
        "array-record",
        "astunparse",
        "attrs",
        "certifi",
        "charset-normalizer",
        "chex",
        "cloudpickle",
        "clu",
        "contourpy",
        "cycler",
        "decorator",
        "distrax",
        "dm-tree",
        "docstring-parser",
        "editdistance",
        "einops",
        "etils",
        "faiss-cpu",
        "faiss-gpu",
        "filelock",
        "flatbuffers",
        "flax",
        "fonttools",
        "fsspec",
        "gast",
        "google-pasta",
        "grpcio",
        "h5py",
        "huggingface-hub",
        "humanize",
        "idna",
        "immutabledict",
        "importlib-resources",
        "jax",
        "jax-cuda12-pjrt",
        "jax-cuda12-plugin",
        "jaxlib",
        "jinja2",
        "keras",
        "kiwisolver",
        "libclang",
        "markdown",
        "markdown-it-py",
        "markupsafe",
        "matplotlib",
        "mdurl",
        "ml-collections",
        "ml-dtypes",
        "mpmath",
        "msgpack",
        "namex",
        "nest-asyncio",
        "networkx",
        "numpy",
        "opt-einsum",
        "optax",
        "optree",
        "orbax-checkpoint",
        "overrides",
        "packaging",
        "pillow",
        "promise",
        "protobuf",
        "psutil",
        "pyarrow",
        "pycocoevalcap",
        "pycocotools",
        "pygments",
        "pyparsing",
        "python-dateutil",
        "pyyaml",
        "regex",
        "requests",
        "rich",
        "safetensors",
        "scipy",
        "sentencepiece",
        "simple-parsing",
        "simplejson",
        "six",
        "sympy",
        "tensorboard",
        "tensorboard-data-server",
        "tensorflow==2.18.0",
        "tensorflow-cpu",
        "tensorflow-gan",
        "tensorflow-hub",
        "tensorflow-io-gcs-filesystem",
        "tensorflow-metadata",
        "tensorflow-probability",
        "tensorflow-text",
        "tensorstore",
        "termcolor",
        "tf-keras",
        "tfds-nightly",
        "tokenizers",
        "toml",
        "toolz",
        "tqdm",
        "transformers",
        "triton",
        "typing-extensions",
        "urllib3",
        "werkzeug",
        "wikipedia-api",
        "wrapt",
        "zipp", 
        "gsutil",
        "flaxformer @ git+https://github.com/google/flaxformer.git",
        "panopticapi @ git+https://github.com/akolesnikoff/panopticapi.git",
        "clu @ git+https://github.com/google/CommonLoopUtils.git"
    ],
    cmdclass={
        "download_assets": DownloadAssetsCommand,
    },
)
