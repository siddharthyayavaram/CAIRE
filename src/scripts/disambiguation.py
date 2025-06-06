import pickle
import numpy as np
from tqdm import tqdm
import torch
from src.utils import save_pickle
from src.config import OUTPUT_PATH, DATA_PATH, LEMMA_EMBEDS
import logging
from pathlib import Path

def lemma_match(args):

    image_paths = sorted(args.image_paths)

    with open(Path(OUTPUT_PATH) / f"{args.timestamp}" / "image_embeddings.pkl", "rb") as f:
        image_embeddings = pickle.load(f)

    with open(Path(OUTPUT_PATH) / f"{args.timestamp}" / "bids_match.pkl", "rb") as f:
        bids_match = pickle.load(f)

    with open(Path(DATA_PATH) / LEMMA_EMBEDS, "rb") as f:
        lemma_embeds = pickle.load(f)

    LEMMA_RESULTS = []

    for i in tqdm(range(len(image_paths))):
        try:
            filename = image_paths[i]

            zimg = np.array(image_embeddings[filename])
            zimg = np.expand_dims(zimg, axis=0)

            bids = []
            for k in bids_match[i]:
                bids.extend(k[0])
            bids = list(set(bids))

            final_results_lemma = []
            batch_size = 64
            num_batches = (len(bids) + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                bids_batch = bids[batch_idx * batch_size:(batch_idx + 1) * batch_size]

                ztxt = np.array([lemma_embeds[bid] for bid in bids_batch])

                # Compute sigmoid of dot products
                scores = zimg @ ztxt.T
                probs = 1.0 / (1.0 + np.exp(-scores))

                # Collect sorted results for this image
                for prob_vector in probs:
                    result = [
                        {"score": float(score), "bid": bid}
                        for score, bid in sorted(
                            zip(prob_vector, bids_batch), key=lambda x: -x[0]
                        )
                    ]
                    final_results_lemma.extend(result)

            torch.cuda.empty_cache()
            
            final_results_lemma = sorted(final_results_lemma, key=lambda x: -x["score"])
            LEMMA_RESULTS.append(final_results_lemma)

        except Exception as e:
            logging.error(f"Error for image {image_paths[i]}: {e}", exc_info=True)

    save_pickle(
        Path(OUTPUT_PATH) / f"{args.timestamp}" / "lemma_match.pkl",
        LEMMA_RESULTS,
        "Lemma matching results"
    )
