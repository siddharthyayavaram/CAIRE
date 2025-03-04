import logging
from src.config import *
from src.utils import load_model
from src.scripts.retrieval import process_images
from src.scripts.disambiguation import lemma_match
from src.scripts.fetch_wikipedia import wiki_ret
from src.scripts.culture_scores import qwen_vl_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_pipeline():
    logging.info("Starting the pipeline...")

    try:
        logging.info("Initializing model...")
        model, params, pp_img, _ = load_model(VARIANT, RES, CKPT_PATH, SEQLEN)

        logging.info("Processing images...")
        process_images(BP, DATASET, model, params, pp_img)

        logging.info("Performing lemma matching...")
        lemma_match(BP, DATASET)

        logging.info("Fetching Wikipedia data...")
        wiki_ret(BP, DATASET, MAX_WIKI_DOCS)

        logging.info("Computing scores with Qwen VL...")
        qwen_vl_scores(BP, DATASET, TARGET_LIST)

        logging.info("Pipeline completed successfully.")

    except Exception:
        logging.error(f"ERROR: ", exc_info=True)

if __name__ == "__main__":
    run_pipeline()
