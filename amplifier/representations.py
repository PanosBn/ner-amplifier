import logging

import spacy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Token:
    def __init__(self, **attributes):
        self.attributes = attributes

    def get_attribute(self, attr):
        return self.attributes.get(attr, None)

    def set_attribute(self, attr, value):
        self.attributes[attr] = value

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
    def __init__(self, file_path: str, column_mapping: dict):
        self.sentences = []
        self.column_mapping = column_mapping
        self.nlp = spacy.load("en_core_web_md")
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
                sentence.add_token(Token(**token_data))
            self._add_pos_tags(sentence)  # Add POS tags if missing
            if sentence.tokens:
                self.add_sentence(sentence)

        logging.info(f"{len(self.sentences)} sentences loaded into the corpus.")

    def _add_pos_tags(self, sentence):
        # Check if POS tags are needed
        needs_pos = any(token.get_attribute("pos") is None for token in sentence.tokens)
        if needs_pos:
            doc = self.nlp(sentence.__str__())
            for token, spacy_token in zip(sentence.tokens, doc):
                if token.get_attribute("pos") is None:
                    token.set_attribute("pos", spacy_token.pos_)

    def filter_empty_sentences(self):
        self.sentences = [sentence for sentence in self.sentences if len(sentence) > 0]

    def __repr__(self):
        return f"Corpus(sentences={self.sentences})"

    def __len__(self):
        return len(self.sentences)
