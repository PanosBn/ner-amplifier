import csv
import logging
from dataclasses import dataclass

import spacy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Token:
    def __init__(self, **attributes):
        self.attributes = attributes
        self.attributes["augmented"] = self.attributes.get("word")

    def get_attribute(self, attr):
        return self.attributes.get(attr, None)

    def set_attribute(self, attr, value):
        self.attributes[attr] = value

    def set_augmented(self, augmented_word):
        self.attributes["augmented"] = augmented_word

    def get_ner_tag(self):
        return self.get_attribute("ner")

    def __str__(self):
        return self.get_attribute("word")

    def __len__(self):
        return len(self.get_attribute("word"))

    def __repr__(self):
        return f"Token({self.attributes})"


class Sentence:
    def __init__(self):
        self.tokens = []

    def add_token(self, token):
        self.tokens.append(token)

    def __str__(self):
        reconstructed_sentence = ""
        for i, token in enumerate(self.tokens):
            if i > 0 and not token.get_attribute("word").startswith(
                ("'", '"', ".", ",", ";", ":", "!", "?", "-", "”", "“", ")", "]", "}")
            ):
                reconstructed_sentence += " "
            reconstructed_sentence += token.get_attribute("word")
        return f"{reconstructed_sentence}"

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __repr__(self):
        return f"Sentence(tokens={self.tokens})"


class Corpus:
    """
    A Corpus object represents a collection of sentences.

    Args:
        file_path (str): The path to the file containing the corpus data.
        column_mapping (dict): A dictionary mapping attribute names to column indices in the file.

    Attributes:
        sentences (list): A list of Sentence objects representing the sentences in the corpus.
        column_mapping (dict): A dictionary mapping attribute names to column indices.
        nlp: A spaCy language model for natural language processing.
    """

    def __init__(self, file_path: str, column_mapping: dict):
        self.sentences = []
        self.column_mapping = column_mapping
        self.nlp = spacy.load("en_core_web_md")
        self._entity_index = {}
        if file_path:
            self._load_data(file_path)

    def _load_data(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.read().split("\n\n")

        for line in lines:
            sentence = Sentence()
            for token_line in line.split("\n"):
                if token_line.strip() == "" or token_line.startswith("-DOCSTART-"):
                    continue
                columns = token_line.split()
                token_data = {
                    attr: columns[idx] if idx < len(columns) else None
                    for attr, idx in self.column_mapping.items()
                }
                token = Token(**token_data)
                sentence.add_token(token)
                self._update_entity_index(token)
            self._add_pos_tags(sentence)  # Add POS tags if missing
            if sentence.tokens:
                self.add_sentence(sentence)

        logging.info(f"{len(self.sentences)} sentences loaded into the corpus.")

    def _update_entity_index(self, token: Token):
        ner_tag = token.get_attribute("ner")
        if ner_tag and ner_tag != "O":
            if ner_tag not in self._entity_index:
                self._entity_index[ner_tag] = 0
            self._entity_index[ner_tag] += 1

    def get_entity_dictionary(self):
        return self._entity_index

    def _add_pos_tags(self, sentence):
        # Check if POS tags are needed
        needs_pos = any(token.get_attribute("pos") is None for token in sentence.tokens)
        if needs_pos:
            doc = self.nlp(sentence.__str__())
            for token, spacy_token in zip(sentence.tokens, doc):
                if token.get_attribute("pos") is None:
                    token.set_attribute("pos", spacy_token.pos_)

    def add_sentence(self, sentence):
        self.sentences.append(sentence)

    def filter_empty_sentences(self):
        self.sentences = [sentence for sentence in self.sentences if len(sentence) > 0]

    def export_to_conll(self, file_path: str, delimiter="\t"):
        """
        Exports the corpus to a file in CONLL format with original tokens replaced by augmented ones.

        Args:
            file_path (str): The path of the file to write to.
            delimiter (str, optional): The delimiter to use in the file. Defaults to '\t' for TSV.

        Returns:
            None
        """
        with open(file_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=delimiter)

            for sentence in self.sentences:
                for token in sentence.tokens:
                    augmented_word = token.get_attribute("augmented")
                    if augmented_word:
                        token.set_attribute("word", augmented_word)

                    row = [
                        token.get_attribute(attr) for attr in self.column_mapping.keys()
                    ]
                    writer.writerow(row)

                writer.writerow([])

    def __repr__(self):
        return f"Corpus(sentences={self.sentences})"

    def __len__(self):
        return len(self.sentences)
