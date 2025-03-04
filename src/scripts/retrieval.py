import numpy as np
import logging
from tqdm import tqdm
from PIL import Image, ImageFile
import torch
from pathlib import Path
from src.utils import get_image_paths, load_faiss_index, load_index_info, save_pickle
from src.config import RETRIEVAL_BATCH_SIZE, NUMBER_RETRIEVED_IMAGES, DATA_PATH, OUTPUT_PATH, INDEX_INFOS, FAISS_INDICES

ImageFile.LOAD_TRUNCATED_IMAGES = True

def process_images(bp, dataset, model, params, pp_img):

    index_info_path = Path(DATA_PATH) / INDEX_INFOS
    faiss_index_path = Path(DATA_PATH) / FAISS_INDICES
    
    urls, ids = load_index_info(index_info_path)
    ind = load_faiss_index(faiss_index_path)

    folder_path = Path(bp) / dataset
    image_paths = sorted(get_image_paths(folder_path))

    batch_size = RETRIEVAL_BATCH_SIZE
    retrieval_count = NUMBER_RETRIEVED_IMAGES  # Number of entities to retrieve
    bids = []
    image_embeddings = {}

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i + batch_size]

        try:
            # Load and preprocess images
            images = [Image.open(filename).convert("RGB") for filename in batch_paths]
            imgs = np.array([pp_img({"image": np.array(image)})["image"] for image in images])

            # Generate embeddings
            zimg, _, _ = model.apply({"params": params}, imgs, None)
            zimg = np.array(zimg)

            # FAISS search
            distances, indices = ind.search(zimg, retrieval_count)

            # Store results
            for j, filename in enumerate(batch_paths):
                image_embeddings[filename] = zimg[j]
                nearest_neighbors = [
                    [ids[i_val], d_val, urls[i_val]] for d_val, i_val in zip(distances[j], indices[j])
                ]
                bids.append(nearest_neighbors)

            del images, imgs, zimg, distances, indices

            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Error processing batch {i // batch_size}: {e}", exc_info=True)

    save_pickle(Path(OUTPUT_PATH) / f"{dataset}_bids_match.pkl", bids, "Bids match data")
    save_pickle(Path(OUTPUT_PATH) / f"{dataset}_image_embeddings.pkl", image_embeddings, "Image embeddings")

