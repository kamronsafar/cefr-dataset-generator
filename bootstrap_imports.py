import subprocess
import sys

# Auto-installer function
def install_and_import(package_name, import_name=None):
    try:
        if import_name:
            globals()[import_name] = __import__(import_name)
        else:
            globals()[package_name] = __import__(package_name)
    except ImportError:
        print(f"📦 '{package_name}' topilmadi. O‘rnatilmoqda...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        if import_name:
            globals()[import_name] = __import__(import_name)
        else:
            globals()[package_name] = __import__(package_name)

# 🔃 Bootstrap all packages
def bootstrap_all():
    install_and_import("tqdm")
    install_and_import("nltk")
    install_and_import("cefrpy")
    install_and_import("requests")

    # built-in modules
    import csv, gc, json, pickle, logging
    import sys
    from multiprocessing import Pool, cpu_count
    from contextlib import contextmanager
    from typing import Optional, List, Tuple

    globals().update({
        "csv": csv,
        "gc": gc,
        "json": json,
        "pickle": pickle,
        "logging": logging,
        "sys": sys,
        "Pool": Pool,
        "cpu_count": cpu_count,
        "contextmanager": contextmanager,
        "Optional": Optional,
        "List": List,
        "Tuple": Tuple
    })

    # NLTK-specific imports
    import nltk
    from nltk.corpus import wordnet, words, brown, gutenberg, webtext, reuters, \
        cess_cat, conll2000, genesis, nps_chat, treebank, inaugural, \
        movie_reviews, state_union, abc
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag, download

    globals().update({
        "nltk": nltk,
        "wordnet": wordnet,
        "words": words,
        "brown": brown,
        "gutenberg": gutenberg,
        "webtext": webtext,
        "reuters": reuters,
        "cess_cat": cess_cat,
        "conll2000": conll2000,
        "genesis": genesis,
        "nps_chat": nps_chat,
        "treebank": treebank,
        "inaugural": inaugural,
        "movie_reviews": movie_reviews,
        "state_union": state_union,
        "abc": abc,
        "WordNetLemmatizer": WordNetLemmatizer,
        "pos_tag": pos_tag,
        "download": download,
    })

