import logging
import os
from datetime import datetime
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.scripts.retrieval import process_images
from src.scripts.disambiguation import lemma_match
from src.scripts.fetch_wikipedia import wiki_retrieval
from src.scripts.culture_scores import qwen_vl_scores
from src.utils import load_model, parse_args, resolve_image_paths, resolve_target_list, log_run_metadata, save_readable
from src.config import MAX_WIKI_DOCS, OUTPUT_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_pipeline(args):

    try:
        logging.info(f"Timestamp: {args.timestamp}")

        logging.info("Initializing model...")
        model, processor = load_model()

        logging.info("Processing images...")
        process_images(args, model, processor)

        logging.info("Performing lemma matching...")
        lemma_match(args)

        logging.info("Fetching Wikipedia data...")
        wiki_retrieval(args, MAX_WIKI_DOCS)

        logging.info("1-5 Scoring...")
        qwen_vl_scores(args)

    except Exception:
        logging.error("ERROR: ", exc_info=True)
        return

if __name__ == "__main__":
    args = parse_args()
    args.target_list, args.is_predefined_list = resolve_target_list(args.target_list)
    args.image_paths, args.is_folder = resolve_image_paths(args)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_folder = Path(OUTPUT_PATH) / timestamp
    run_folder.mkdir(parents=True, exist_ok=True)

    args.timestamp = timestamp
    log_run_metadata(args) 
    run_pipeline(args)
    save_readable(args, OUTPUT_PATH)
    logging.info(f"Outputs: {Path(OUTPUT_PATH) / f'{args.timestamp}' / 'combined_outputs.csv'}")
