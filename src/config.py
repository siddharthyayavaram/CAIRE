from pathlib import Path

DEFAULT_DATASET = Path("src") / "examples"
DATA_PATH = Path("data")
# Use relative path for outputs
OUTPUT_PATH = Path("src") / "outputs"

PREDEFINED_TARGET_LISTS = [
    DATA_PATH / "country_list.pkl",         
    DATA_PATH / "top10_countries.pkl",      
    DATA_PATH / "indian_states.pkl",        
    DATA_PATH / "USA_states.pkl",           
    DATA_PATH / "common_religions.pkl",     
]

INDEX_INFOS = "index_infos_merged.json"
FAISS_INDICES = "faiss_index_merged"
LEMMA_EMBEDS = "combined_lemma_embeds.pkl"
BABELNET_WIKI = 'babelnet_source_dict.pkl'

RETRIEVAL_BATCH_SIZE = 64
NUMBER_RETRIEVED_IMAGES = 20

MAX_WIKI_DOCS = 10
USE_MULTIPLE_WIKI_PAGES = False  # Set to True to use multiple Wikipedia pages in VLM scoring
WIKI_CHARS_PER_PAGE = 2000  # Characters to use per Wikipedia page when USE_MULTIPLE_WIKI_PAGES is True

PROMPT_TEMPLATE = '''We want to assess how relevant an image is to a given culture. 
We have identified this concept to be closely associated with the image: {entity}. 
Here is some detailed information about this concept from Wikipedia. {wiki}.

Assign a score from 1 to 5 based on how culturally relevant the image is to {target}:
Think step by step, specifically considering cultural symbols, styles, traditions, or any features that align with the culture of {target}.
The final score should be a number between 1 to 5, where 1 and 5 mean the following:
1 -- Not relevant: – The content does not connect with or reflect the target culture at all.
2 -- Minimally Relevant: – The content shows slight or superficial connections to the culture but lacks depth. May include vague references or isolated cultural elements that feel out of place or underdeveloped.
3 -- Somewhat Relevant: – The content contains identifiable cultural references, but they may feel generic, inconsistent, or limited in scope. The connection to the culture is present but could be stronger or more meaningful.
4 -- Relevant: – The content reflects a reasonable understanding of the culture, including accurate and appropriate references. It integrates cultural aspects well, though there may still be areas where more depth could be added.
5 -- Highly Relevant: – The content is deeply connected to the target culture, showing an immersive, accurate, and respectful understanding. Cultural references feel natural, meaningful, and central to the content.

In your reasoning process, please consider BOTH the image and the information about the entity provided above.

Provide your response in the following JSON format:
{{"score": <number between 1-5>, "reasoning": "<your detailed reasoning>"}}
'''

PROMPT_TEMPLATE_MULTI = '''We want to assess how relevant an image is to a given culture. 
We have identified several related concepts that are closely associated with the image. These Wikipedia titles are listed in decreasing order of relevance: {entities}.
Here is some detailed information about these concepts from Wikipedia:

{wiki}


Assign a score from 1 to 5 based on how culturally relevant the image is to {target}:
Think step by step, specifically considering cultural symbols, styles, traditions, or any features that align with the culture of {target}.

The final score should be a number between 1 to 5, where 1 and 5 mean the following:
1 -- Not relevant: – The content does not connect with or reflect the target culture at all.
2 -- Minimally Relevant: – The content shows slight or superficial connections to the culture but lacks depth. May include vague references or isolated cultural elements that feel out of place or underdeveloped.
3 -- Somewhat Relevant: – The content contains identifiable cultural references, but they may feel generic, inconsistent, or limited in scope. The connection to the culture is present but could be stronger or more meaningful.
4 -- Relevant: – The content reflects a reasonable understanding of the culture, including accurate and appropriate references. It integrates cultural aspects well, though there may still be areas where more depth could be added.
5 -- Highly Relevant: – The content is deeply connected to the target culture, showing an immersive, accurate, and respectful understanding. Cultural references feel natural, meaningful, and central to the content.

In your reasoning process, please consider BOTH the image and the information about these entities provided above. Note that the first entity is most relevant to the image, while others provide additional context.

Provide your response in the following JSON format:
{{"score": <number between 1-5>, "reasoning": "<your detailed reasoning>"}}
'''