import pickle
import jax
import jax.numpy as jnp
from tqdm import tqdm
import torch
from src.utils import get_image_paths, save_pickle
from src.config import OUTPUT_PATH, DATA_PATH, LEMMA_EMBEDS
import logging
from pathlib import Path

def lemma_match(bp, dataset):

    folder_path = Path(bp) / dataset
    image_paths = sorted(get_image_paths(folder_path))

    with open(Path(OUTPUT_PATH) / f"{dataset}_image_embeddings.pkl", "rb") as f:
        image_embeddings = pickle.load(f)

    with open(Path(OUTPUT_PATH) / f"{dataset}_bids_match.pkl", "rb") as f:
        bids_match = pickle.load(f)

    with open(Path(DATA_PATH) / LEMMA_EMBEDS, "rb") as f:
        lemma_embeds = pickle.load(f)

    LEMMA_RESULTS = []

    for i in tqdm(range(len(image_paths))):
        try:
            filename = image_paths[i]
            if filename.endswith(('.JPEG', '.png', '.jpg','.jpeg')):

                zimg = image_embeddings[filename]
                zimg = jnp.array(zimg)
                zimg = jnp.expand_dims(zimg, axis=0)

                bids = []

                for k in bids_match[i]:
                    bids.extend(k[0])

                bids = list(set(bids))

                final_results_lemma = []

                batch_size = 64
                num_batches = (len(bids) + batch_size - 1) // batch_size

                for batch_idx in range(num_batches):
                    bids_batch = bids[batch_idx * batch_size:(batch_idx + 1) * batch_size]

                    # Retrieve precomputed embeddings for each bid in the batch
                    ztxt = jnp.array([lemma_embeds[bid] for bid in bids_batch])

                    # Calculate probabilities
                    probs = jax.nn.sigmoid(zimg @ ztxt.T)

                    for prob in probs:
                        result = [
                            {"score": score.item(), "bid": bid}
                            for score, bid in sorted(zip(prob, bids_batch), key=lambda x: -x[0])
                        ]

                        final_results_lemma.extend(result)

                    jax.device_put(None, jax.devices("gpu")[0])

                torch.cuda.empty_cache()
                final_results_lemma = sorted(final_results_lemma, key=lambda x: -x["score"])

                LEMMA_RESULTS.append(final_results_lemma)

        except Exception as e:
            logging.error(f"Error for image {image_paths[i]}: {e}", exc_info=True)

    save_pickle(Path(OUTPUT_PATH) / f"caire_{dataset}_lemma_match.pkl", LEMMA_RESULTS, "Lemma matching results")