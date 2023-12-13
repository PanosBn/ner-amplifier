import logging
import random
from pathlib import Path
from typing import Optional, Union

import numpy as np
from nltk.corpus import wordnet
from tqdm import tqdm

from amplifier.representations import Corpus, Sentence

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)


class EntityAugmenter:
    def __init__(self, corpus: Corpus):
        logging.info("Initialized Entity augmenter.")
        if isinstance(corpus, Corpus):
            self.corpus = corpus
        else:
            raise TypeError("Expected a Corpus object.")

    def entity_swap(self, swap_prob: float = 0.5, text: Optional[Sentence] = None):
        if text and isinstance(text, Sentence):
            if random.random() < swap_prob:
                return


#         elif isinstance(text, Corpus):
#             for sentence in tqdm(text.sentences, desc="Swapping entities"):
#                 if random.random() < swap_prob:


#         # raise NotImplementedError


#     def entity_replace(self, text: Sentence | Corpus):
#         raise NotImplementedError
