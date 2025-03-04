import os
import json
import sys
import pickle
import logging
import faiss
import ml_collections  # type: ignore

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(BASE_DIR, "big_vision"))

import big_vision.models.proj.image_text.two_towers as model_mod  # type: ignore
import big_vision.pp.builder as pp_builder  # type: ignore
import big_vision.pp.ops_image # type: ignore
import big_vision.pp.ops_text # type: ignore

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_model(variant, res, ckpt_path, seqlen):
    try:
        if variant.endswith('-i18n'):
            variant = variant[:-len('-i18n')]

        model_cfg = ml_collections.ConfigDict()
        model_cfg.image_model = 'vit'
        model_cfg.text_model = 'proj.image_text.text_transformer'
        model_cfg.image = dict(variant=variant, pool_type='map')
        model_cfg.text = dict(variant=variant.split('/')[0], vocab_size=250_000)
        model_cfg.out_dim = (None, None)
        model_cfg.bias_init = -10.0
        model_cfg.temperature_init = 10.0

        model = model_mod.Model(**model_cfg)
        init_params = None
        params = model_mod.load(init_params, str(ckpt_path), model_cfg)

        pp_img = pp_builder.get_preprocess_fn(f'resize({res})|value_range(-1, 1)')
        pp_txt = pp_builder.get_preprocess_fn(f'tokenize(max_len={seqlen}, model="sentencepiece.model", eos="sticky", pad_value=1, inkey="text")')

        return model, params, pp_img, pp_txt

    except Exception as e:
        logging.error(f"Failed to load model: {e}", exc_info=True)
        return None, None, None, None

def get_image_paths(folder, extensions=("jpg", "jpeg", "png", "gif")):
    image_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(extensions):
                image_files.append(os.path.join(root, file))
    return image_files

def load_index_info(index_info_path):
    try:
        with open(index_info_path, 'r') as file:
            data = json.load(file)
        return data["image_urls"], data["babelnet_ids"]
    except Exception as e:
        logging.error(f"Failed to load index information: {e}", exc_info=True)
        return None, None

def load_faiss_index(index_path):
    try:
        return faiss.read_index(index_path)
    except Exception as e:
        logging.error(f"Failed to load FAISS index: {e}", exc_info=True)
        return None
    
def save_pickle(file_path, data, description):
    try:
        with open(file_path, "wb") as file:
            pickle.dump(data, file)
        logging.info(f"Saved {description} to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save {description}: {e}", exc_info=True)