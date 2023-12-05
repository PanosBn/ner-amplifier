import logging
from pathlib import Path
from typing import Union

import numpy as np
from nltk.corpus import wordnet
from tqdm import tqdm

from amplifier.representations import Corpus, Sentence

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)


class NounAugmenter:
    def __init__(self):
        logging.info(f"Initialized Noun augmenter.")
        pass

    def noun_augment_sense2vec(
        self, text: Sentence | Corpus, model_path: Union[Path, str]
    ):
        """
        Augment nouns in the text with sense2vec.
        :param text: Sentence or Corpus object
        :param model_path: Path to the sense2vec model

        """
        from sense2vec import Sense2Vec

        path = Path(model_path)
        if not path.exists():
            raise ValueError(f"{model_path} path does not exist.")
        s2v = Sense2Vec().from_disk(path)
        if isinstance(text, Sentence):
            self._process_sentence(text, method="sense2vec", model=s2v)
        elif isinstance(text, Corpus):
            for sentence in tqdm(text.sentences, desc="Augmenting with Sense2Vec"):
                self._process_sentence(sentence, method="sense2vec", model=s2v)
        else:
            raise TypeError("Expected a Sentence or Corpus object.")

    def noun_augment_word2vec(
        self, text: Sentence | Corpus, model_path: Union[Path, str]
    ):
        from gensim.models import KeyedVectors

        """
        Augment nouns in the text with word2vec.
        :param text: Sentence or Corpus object
        :param model_path: Path to the word2vec model
        """

        w2v = KeyedVectors.load_word2vec_format(model_path, binary=True)

        if isinstance(text, Sentence):
            self._process_sentence(text, method="word2vec", model=w2v)
        elif isinstance(text, Corpus):
            for sentence in tqdm(text.sentences, desc="Augmenting with Word2Vec"):
                self._process_sentence(sentence, method="word2vec", model=w2v)
        else:
            raise TypeError("Expected a Sentence or Corpus object.")

    def noun_augment_wordnet(self, text: Sentence | Corpus):
        """
        Augment nouns in the text with WordNet.
        :param text: Sentence or Corpus object
        """
        if isinstance(text, Sentence):
            self._process_sentence(text, method="wordnet")
        elif isinstance(text, Corpus):
            for sentence in tqdm(text.sentences, desc="Augmenting with WordNet"):
                self._process_sentence(sentence, method="wordnet")
        else:
            raise TypeError("Expected a Sentence or Corpus object.")

    def _process_sentence(self, sentence: Sentence, method: str, model=None):
        for token in sentence.tokens:
            original_word = token.get_attribute("word")
            pos_tag = token.get_attribute("pos")
            ner_tag = token.get_ner_tag()

            pos_tag = token.get_attribute("pos")
            if (pos_tag == "NOUN" or pos_tag == "PROPN") and ner_tag == "O":
                augmented_word = original_word

                if method == "wordnet":
                    synonyms = [
                        syn.name().split(".")[0]
                        for syn in wordnet.synsets(original_word, pos=wordnet.NOUN)
                    ]
                    single_word_synonyms = [
                        syn for syn in synonyms if "_" not in syn and " " not in syn
                    ]
                    if single_word_synonyms:
                        augmented_word = np.random.choice(single_word_synonyms)

                elif method == "sense2vec":
                    query = str(original_word + "|NOUN")
                    if query in model:
                        most_similar = model.most_similar(query, n=5)
                        single_word_synonyms = [
                            item[0].split("|")[0]
                            for item in most_similar
                            if "_" not in item[0]
                        ]
                        if single_word_synonyms:
                            augmented_word = np.random.choice(single_word_synonyms)

                elif method == "word2vec":
                    if original_word in model.key_to_index:
                        similar_words = model.similar_by_word(original_word, topn=5)
                        single_word_synonyms = [
                            word
                            for word, _ in similar_words
                            if " " not in word and "_" not in word
                        ]
                        if single_word_synonyms:
                            augmented_word = np.random.choice(single_word_synonyms)

                token.set_augmented(augmented_word)
