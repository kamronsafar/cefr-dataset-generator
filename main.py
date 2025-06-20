"""
🔤 CEFR Dataset Generator v2.1
Author: Kamron Safar
Description: Generates a CEFR-labeled English word dataset using multiprocessing, NLTK, and cefrpy.
"""
from bootstrap_imports import bootstrap_all
bootstrap_all()
import os
import csv
import gc
import json
import pickle
import logging
import sys
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm
from cefrpy import CEFRAnalyzer
from nltk.corpus import wordnet
from nltk import download, pos_tag
from nltk.corpus import (words, brown, gutenberg, webtext, reuters,
                         cess_cat, conll2000, genesis, nps_chat, treebank, inaugural,
                         movie_reviews, state_union, abc)
from nltk.stem import WordNetLemmatizer
from typing import Optional, List, Tuple
from contextlib import contextmanager
import requests

# ------------------ CONFIG ------------------ #
CSV_FILENAME = "cefr_mega_dataset.csv"
CACHE_DIR = "cefr_cache"
PROGRESS_FILE = "progress.json"
MAX_WORDS = 2000000

# ------------------ LOGGER ------------------ #
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
lemmatizer = WordNetLemmatizer()

# ------------------ NLTK SETUP ------------------ #
def download_nltk_data():
    for pkg in ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4', 'words',
                'brown', 'gutenberg', 'webtext', 'reuters', 'cess_cat', 'conll2000',
                'genesis', 'nps_chat', 'treebank']:
        download(pkg, quiet=True)

# ------------------ WORD COLLECTOR ------------------ #
def get_all_words() -> List[str]:
    sources = [
        words.words(),
        brown.words(),
        gutenberg.words(),
        webtext.words(),
        reuters.words(),
        cess_cat.words(),
        conll2000.words(),
        genesis.words(),
        nps_chat.words(),
        treebank.words(),
        inaugural.words(),
        movie_reviews.words(),
        state_union.words(),
        abc.words(),
        wordnet.words()
]


    urls = [
        "https://norvig.com/ngrams/word.list",
        "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
    ]

    for url in urls:
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                sources.append(response.text.splitlines())
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch {url}: {e}")

    unique_words = set()
    for word_list in tqdm(sources, desc="📥 Loading word sources"):
        unique_words.update(
            w.lower().strip()
            for w in word_list
            if isinstance(w, str) and w.isalpha() and 2 < len(w) < 25
        )

    return sorted(list(unique_words))[:MAX_WORDS]

# ------------------ CACHE ------------------ #
class DiskCache:
    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.cache_file = os.path.join(CACHE_DIR, "cache.pkl")
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"⚠️ Cache load error: {e}")
        return {}

    def __contains__(self, word):
        return word.lower() in self.cache

    def __getitem__(self, word):
        return self.cache.get(word.lower())

    def __setitem__(self, word, value):
        self.cache[word.lower()] = value

    def save(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)

# ------------------ CEFR ANALYZER WRAPPER ------------------ #
class CEFRAnalyzerWrapper:
    def __init__(self):
        self.analyzer = CEFRAnalyzer()

    def get_cefr_level(self, word: str) -> Optional[str]:
        try:
            return self.analyzer.get_average_word_level_CEFR(word)
        except Exception:
            return None

# ------------------ LEMMATIZER ------------------ #
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('R'):
        return 'r'
    return 'n'

def lemmatize_word(word: str) -> str:
    pos = pos_tag([word])[0][1]
    wn_pos = get_wordnet_pos(pos)
    return lemmatizer.lemmatize(word, wn_pos)

# ------------------ MULTIPROCESSING ------------------ #
analyzer = None

def init_worker():
    global analyzer
    analyzer = CEFRAnalyzerWrapper()

def map_pos_to_cefrpy(nltk_pos: str) -> Optional[str]:
    if nltk_pos.startswith("N"):
        return "noun"
    elif nltk_pos.startswith("V"):
        return "verb"
    elif nltk_pos.startswith("J"):
        return "adj"
    elif nltk_pos.startswith("R"):
        return "adv"
    return None

def process_batch(word: str) -> Optional[Tuple[str, str, str, str]]:
    try:
        cefr = analyzer.analyzer.get_average_word_level_CEFR(word)
        if not cefr:
            return None

        synsets = wordnet.synsets(word)
        definition = synsets[0].definition() if synsets else ""

        synonyms = ', '.join(list({l.name() for s in synsets[:3] for l in s.lemmas()
                                   if l.name().lower() != word.lower()})[:5])

        return (word, cefr, definition, synonyms)

    except Exception as e:
        logger.debug(f"❌ Error in word '{word}': {e}")
        return None




# ------------------ PROGRESS ------------------ #
@contextmanager
def progress_manager():
    try:
        with open(PROGRESS_FILE, 'r') as f:
            progress = json.load(f)
    except Exception:
        progress = {'last_index': 0}
    try:
        yield progress
    finally:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f)

# ------------------ MAIN FUNCTION ------------------ #
def main():
    try:
        download_nltk_data()
        cache = DiskCache()

        with progress_manager() as progress:
            all_words = get_all_words()
            start = progress.get('last_index', 0)

            with open(CSV_FILENAME, 'a' if start > 0 else 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if start == 0:
                    writer.writerow(["Word", "CEFR", "Definition", "Synonyms"])

                batch_size = 1000
                for i in tqdm(range(start, len(all_words), batch_size), desc="🔧 Total Progress"):
                    batch = all_words[i:i + batch_size]
                    with Pool(cpu_count(), initializer=init_worker) as pool:
                        results = list(tqdm(pool.imap(process_batch, batch),
                                            total=len(batch),
                                            desc=f"🔎 Batch {i // batch_size + 1}",
                                            leave=False))
                    valid = [r for r in results if r]
                    for entry in valid:
                        writer.writerow(entry)
                        cache[entry[0]] = entry
                    progress['last_index'] = i + batch_size
                    cache.save()
                    gc.collect()

                logger.info(f"✅ Dataset saved to: {CSV_FILENAME}")
                logger.info(f"🔢 Total words: {len(all_words)}")
                logger.info(f"📈 Valid entries: {len(cache.cache)}")

    except KeyboardInterrupt:
        logger.warning("⛔ Interrupted by user.")
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

