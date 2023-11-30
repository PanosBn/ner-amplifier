import numpy as np
from nltk.corpus import wordnet
from spacy.language import Language

from amplifier import Corpus, Sentence


class Augmenter:
    def __init__(self, nlp: Language):
        if not isinstance(nlp, Language):
            raise TypeError("Expected a SpaCy Language pipeline instance.")

        self.nlp = nlp

    def noun_augment_wordnet(self, text: Sentence | Corpus):
        if isinstance(text, Sentence):
            self._process_sentence(text)
        elif isinstance(text, Corpus):
            for sentence in text.sentences:
                self._process_sentence(sentence)
        else:
            raise TypeError("Expected a Sentence or Corpus object.")

    def _process_sentence(self, sentence: Sentence):
        doc = self.nlp(f"{sentence}")

        if len(doc) != len(sentence.tokens):
            raise ValueError(
                "The token count from SpaCy does not match the original sentence."
            )

        for original_token, spacy_token in zip(sentence.tokens, doc):
            if spacy_token.pos_ == "NOUN":
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
