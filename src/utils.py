import os
import csv
import json
import faiss
import pickle
import logging
import argparse
from pathlib import Path
from src import config as cfg
from transformers import AutoProcessor, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_model():
    try:
        model = AutoModel.from_pretrained("google/siglip-so400m-patch16-256-i18n")
        processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch16-256-i18n")
        return model, processor
    
    except Exception as e:
        logging.error(f"Failed to load model: {e}", exc_info=True)
        return None, None

def get_image_paths(folder, extensions=("jpg", "jpeg", "png", "gif")):
    image_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(extensions):
                image_files.append(os.path.join(root, file))
    return image_files

def load_index_info(index_info_path):
    try:
        with open(str(index_info_path), 'r') as file:
            data = json.load(file)
        return data["image_urls"], data["babelnet_ids"]
    except Exception as e:
        logging.error(f"Failed to load index information: {e}", exc_info=True)
        return None, None

def load_faiss_index(index_path):
    try:
        return faiss.read_index(str(index_path))
    except Exception as e:
        logging.error(f"Failed to load FAISS index: {e}", exc_info=True)
        return None
    
def save_pickle(file_path, data, description):
    try:
        with open(str(file_path), "wb") as file:
            pickle.dump(data, file)
        logging.info(f"Saved {description} to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save {description}: {e}", exc_info=True)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target_list",
        type=lambda s: [item.strip() for item in s.split(",")],
        default=["top10_countries.pkl"],
        help="Comma-separated list of target cultures"
    )

    parser.add_argument(
        "--image_paths",
        nargs="+",
        help="List of image paths to process"
    )
    return parser.parse_args()

def resolve_target_list(arg):
    path = cfg.DATA_PATH / arg[0]
    if len(arg) == 1 and path in cfg.PREDEFINED_TARGET_LISTS:
        with open(path, "rb") as f:
            return pickle.load(f), arg[0]
    return arg, False

def resolve_image_paths(args):
    if args.image_paths:
        if len(args.image_paths) == 1 and os.path.isdir(args.image_paths[0]):
            return sorted([Path(p) for p in get_image_paths(args.image_paths[0])]), True

        valid_paths = [Path(p) for p in args.image_paths if os.path.isfile(p)]
        if not valid_paths:
            raise FileNotFoundError("Invalid file paths")
        return sorted(valid_paths), False

    default_folder = cfg.DEFAULT_DATASET
    return sorted([Path(p) for p in get_image_paths(default_folder)]), True

def log_run_metadata(args, log_file="run_log.csv"):
    log_path = Path(cfg.OUTPUT_PATH) / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if args.is_folder:
        folder_name = args.image_paths[0].parent.name

    if args.is_predefined_list:
        target_info = args.is_predefined_list
    else:
        target_info = ",".join(args.target_list)

    row = {
        "timestamp": args.timestamp,
        "image_input_type": "folder" if args.is_folder else "list",
        "num_images": len(args.image_paths),
        "image_paths": folder_name if args.is_folder else " ".join([str(p) for p in args.image_paths]),
        "targets": target_info
    }

    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)