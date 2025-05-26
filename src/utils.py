import os
import json
import pickle
import logging
import faiss
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