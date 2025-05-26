import logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.config import *
from src.utils import load_model
from src.scripts.retrieval import process_images
from src.scripts.disambiguation import lemma_match
from src.scripts.fetch_wikipedia import wiki_ret

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_pipeline():
    logging.info("Starting VEL...")

    try:
        logging.info("Initializing model...")
        model, processor = load_model()

        logging.info("Processing images...")
        process_images(BP, DATASET, model, processor)

        logging.info("Performing lemma matching...")
        lemma_match(BP, DATASET)

        logging.info("Fetching Wikipedia data...")
        wiki_ret(DATASET, MAX_WIKI_DOCS)

        logging.info("VEL completed successfully.")

    except Exception:
        logging.error(f"ERROR: ", exc_info=True)
        return

if __name__ == "__main__":
    run_pipeline()
