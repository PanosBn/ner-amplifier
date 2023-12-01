import logging
from pathlib import Path
from typing import Union

import numpy as np
import spacy_conll
from nltk.corpus import wordnet
from sense2vec import Sense2Vec
from spacy.language import Language

from amplifier.representations import Corpus, Sentence

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class NounAugmenter:
    def __init__(self, nlp: Language):
        if not isinstance(nlp, Language):
            raise TypeError("Expected a SpaCy Language pipeline instance.")

        self.nlp = nlp

    def noun_augment_sense2vec(
        self, text: Sentence | Corpus, model_path: Union[Path, str]
    ):
        """
        Augment nouns in a sentence or corpus using sense2vec.
        To use this method, you must first download the sense2vec model from https://github.com/explosion/sense2vec
        """
        path = Path(model_path)
        self.s2v = Sense2Vec().from_disk(path)

        if not path.exists():
            raise ValueError(f"{model_path} path does not exist.")
        if isinstance(text, Sentence):
            self._process_sentence(text, method="sense2vec")
        elif isinstance(text, Corpus):
            for sentence in text.sentences:
                self._process_sentence(sentence)
        else:
            raise TypeError("Expected a Sentence or Corpus object.")

    def noun_augment_wordnet(self, text: Sentence | Corpus):
        if isinstance(text, Sentence):
            self._process_sentence(text, method="wordnet")
        elif isinstance(text, Corpus):
            for sentence in text.sentences:
                self._process_sentence(sentence, method="wordnet")
        else:
            raise TypeError("Expected a Sentence or Corpus object.")

    def noun_augment_word2vec(
        self, text: Sentence | Corpus, model_path: Union[Path, str]
    ):
        """
        Augment nouns in a sentence or corpus using word2vec.
        To use this method, you must first download a word2vec model.
        """
        path = Path(model_path)
        if not path.exists():
            raise ValueError(f"{model_path} path does not exist.")
        raise NotImplementedError

    def _process_sentence(self, sentence: Sentence, method: str):
        doc = self.nlp(f"{sentence}")

        if len(doc) != len(sentence.tokens):
            logging.info(
                f"Original sentence: {sentence} length: {len(sentence.tokens)}"
            )
            logging.info(f"SpaCy sentence: {doc} length: {len(doc)}")
            raise ValueError(
                "The token count from SpaCy does not match the original sentence."
            )

        for original_token, spacy_token in zip(sentence.tokens, doc):
            if spacy_token.pos_ == "NOUN" and original_token.get_ner_tag() == "O":
                if method == "wordnet":
                    synonyms = [
                        syn.name().split(".")[0]
                        for syn in wordnet.synsets(spacy_token.text, pos=wordnet.NOUN)
                    ]
                    one_word_synonyms = [
                        syn for syn in synonyms if "_" not in syn and " " not in syn
                    ]

                    if one_word_synonyms:
                        synonym = np.random.choice(one_word_synonyms)
                        original_token.set_word(synonym)
                elif method == "sense2vec":
                    query = str(spacy_token.text + "|NOUN")
                    logging.info(f"Querying sense2vec for {query}")
                    assert query in self.s2v
                    most_similar = self.s2v.most_similar(query, n=5)
                    logging.info(f"Most similar words: {most_similar}")
                    filtered_synonyms = [
                        item[0].split("|")[0]
                        for item in most_similar
                        if "_" not in item[0]
                    ]
                    if len(filtered_synonyms) >= 1:
                        chosen_synonym = np.random.choice(filtered_synonyms)
                        original_token.set_word(chosen_synonym)
                    else:
                        logging.info(f"No synonyms found for {query}")
