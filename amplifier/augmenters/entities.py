import logging
import random
from pathlib import Path
from typing import Union

import numpy as np
from nltk.corpus import wordnet
from tqdm import tqdm

from amplifier.representations import Corpus, Sentence

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)

# class EntityAugmenter():
#     def __init__(self):
#         pass

#     def entity_swap(self, text: Sentence | Corpus, swap_prob: float = 0.5):
#         if isinstance(text, Sentence):
#             if random.random() < swap_prob:


#         elif isinstance(text, Corpus):
#             for sentence in tqdm(text.sentences, desc="Swapping entities"):
#                 if random.random() < swap_prob:


#         # raise NotImplementedError


#     def entity_replace(self, text: Sentence | Corpus):
#         raise NotImplementedError
