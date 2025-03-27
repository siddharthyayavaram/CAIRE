import logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.config import *
from src.scripts.culture_scores import qwen_vl_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_pipeline():
    logging.info("Starting cultural relevance scoring...")

    try:
        logging.info("Computing scores with Qwen VL...")
        qwen_vl_scores(BP, DATASET, TARGET_LIST)

        logging.info("Scoring completed successfully.")

    except Exception:
        logging.error(f"ERROR: ", exc_info=True)
        return

if __name__ == "__main__":
    run_pipeline()
