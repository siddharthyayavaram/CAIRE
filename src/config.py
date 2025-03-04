from pathlib import Path

BP = Path(".")
DATASET = 'examples'  # Dataset name
DATA_PATH = Path("data")
OUTPUT_PATH = Path("outputs")
TARGET_LIST = Path(DATA_PATH) / "country_list.pkl"

INDEX_INFOS = "index_infos_merged"
FAISS_INDICES = "faiss_index_merged"
LEMMA_EMBEDS = "combined_lemma_embeds.pkl"
BABELNET_WIKI = 'babelnet_source_dict.pkl'

RETRIEVAL_BATCH_SIZE = 64
NUMBER_RETRIEVED_IMAGES = 20

VARIANT = 'So400m/16-i18n'
RES = 256
SEQLEN = 64
CKPT_PATH = Path("checkpoints") / "webli_i18n_so400m_16_256_78061115.npz"  # model ckpt path

MAX_WIKI_DOCS = 20

PROMPT_TEMPLATE = '''We want to assess how relevant an image is to a given culture. 
We have identified this concept to be closely associated with the image: {entity}. 
Here is some detailed information about this concept from Wikipedia. {wiki}.
Using the above context, assign a score from 1 to 5 based on how culturally relevant the image is to {target}:
Think step by step, specifically considering cultural symbols, styles, traditions, or any features that align with the culture of {target}.
The final score should be a number between 1 to 5, where 1 and 5 mean the following:
1 -- Not relevant: – The content does not connect with or reflect the target culture at all.
2 -- Minimally Relevant: – The content shows slight or superficial connections to the culture but lacks depth. May include vague references or isolated cultural elements that feel out of place or underdeveloped.
3 -- Somewhat Relevant: – The content contains identifiable cultural references, but they may feel generic, inconsistent, or limited in scope. The connection to the culture is present but could be stronger or more meaningful.
4 -- Relevant: – The content reflects a reasonable understanding of the culture, including accurate and appropriate references. It integrates cultural aspects well, though there may still be areas where more depth could be added.
5 -- Highly Relevant: – The content is deeply connected to the target culture, showing an immersive, accurate, and respectful understanding. Cultural references feel natural, meaningful, and central to the content.
The output should be a single number ONLY:
'''
