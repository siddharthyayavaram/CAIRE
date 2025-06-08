import pickle
import random
import wikipediaapi
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from src.config import OUTPUT_PATH, DATA_PATH, BABELNET_WIKI
from src.utils import save_pickle
import logging

logging.getLogger("wikipediaapi").setLevel(logging.CRITICAL)
USER_AGENTS = [
    "MyPythonScript/1.0 (contact@example.com)",
    "MyPythonScript/2.0 (contact2@example.com)",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
]

def toeng(language, page_title):
    user_agent = random.choice(USER_AGENTS)
    try:
        wiki = wikipediaapi.Wikipedia(language=language, user_agent=user_agent)
        page = wiki.page(page_title)
        if page.exists():
            if language != 'en':
                lang_links = page.langlinks
                if 'en' in lang_links:
                    return lang_links['en'].title
        return False
    except:
        return False

def fetch_wikipedia_page(page_title):
    try:
        user_agent = random.choice(USER_AGENTS)
        wiki = wikipediaapi.Wikipedia(language='en', user_agent=user_agent)
        page = wiki.page(page_title)
        return {
            "title": page.title,
            "summary": page.summary,
            "page_id": page.pageid,
            "text": page.text,
            "categories": list(page.categories.keys()),
            "sections": [section.title for section in page.sections]
        }
    except:
        return {'text': ""}

def get_wiki_page(title):
    p = fetch_wikipedia_page(title)
    return p if p and len(p['text']) > 10 else None

def get_en_pages(titles):
    for title, lang in titles:
        if lang == 'EN':
            p = get_wiki_page(title)
            if p:
                return p
    return None

def get_non_en_pages(titles):
    for title, lang in titles:
        if lang != 'EN':
            translated = toeng(lang.lower(), title)
            if translated:
                p = get_wiki_page(translated)
                if p:
                    return p
    return None

def process_group_bids(group_bids, babelnet_dict, max_docs):
    group_pages = []
    for bid in group_bids:
        if len(group_pages) >= max_docs:
            break

        wiki_direct, wiki_redirect = babelnet_dict[bid]
        all_wiki = wiki_direct + wiki_redirect

        p = get_en_pages(all_wiki)
        if p:
            group_pages.append(p)
            continue

        p = get_non_en_pages(all_wiki)
        if p:
            group_pages.append(p)

    return group_pages

def wiki_retrieval(args, max_docs=20):
    with open(Path(DATA_PATH) / BABELNET_WIKI, 'rb') as f:
        babelnet_dict = pickle.load(f)

    with open(Path(OUTPUT_PATH) / f'{args.timestamp}' / "lemma_match.pkl", 'rb') as f:   
        y = pickle.load(f)

    all_bids = [[j['bid'] for j in i] for i in y[:]]

    def process(bid_group):
        return process_group_bids(bid_group, babelnet_dict, max_docs)

    with ThreadPoolExecutor(max_workers=16) as executor:
        outputs = list(tqdm(
            executor.map(process, all_bids),
            total=len(all_bids)
        ))

    save_pickle(Path(OUTPUT_PATH) / f"{args.timestamp}" / "WIKI.pkl", outputs, "Wikipedia content")
